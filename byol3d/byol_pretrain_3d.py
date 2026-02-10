"""
BYOL 3D pretraining loop.

Mirrors byol_experiment.py from the original DeepMind repo, reimplemented
in PyTorch for the 3D lightsheet domain.

Training procedure (paper Appendix J):
  1. Sample batch of 256^3 patches
  2. Extract two 96^3 crops with overlap >= 0.4
  3. Apply augmentations independently to each view
  4. Forward both views through online network -> predictions
  5. Forward both views through target network -> projections (no grad)
  6. Compute symmetrized regression loss (Eq. 2)
  7. Update online params with AdamW
  8. Update target params with EMA (Eq. 1)

Usage:
    # Full training on cluster:
    python -m byol.byol_pretrain_3d --config full

    # Smoke test locally:
    python -m byol.byol_pretrain_3d --config smoke
"""

from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn

from byol3d.configs.lightsheet_3d import get_config, get_smoke_test_config
from byol3d.utils.networks_3d import (
    create_byol_pair,
    regression_loss,
    update_target_ema,
    cosine_ema_schedule,
)
from byol3d.utils.dataset_blosc2 import create_byol_dataloader


# ---------------------------------------------------------------------------
# Learning rate schedule (mirrors schedules.py from original repo)
# ---------------------------------------------------------------------------

def cosine_lr_schedule(
    step: int,
    max_steps: int,
    base_lr: float,
    warmup_steps: int = 0,
) -> float:
    """Cosine LR schedule with linear warmup.

    Matches the original repo's learning_schedule function.
    """
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * step / warmup_steps

    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(progress, 1.0)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Single training step
# ---------------------------------------------------------------------------

def train_step(
    online: nn.Module,
    target: nn.Module,
    optimizer: torch.optim.Optimizer,
    view1: torch.Tensor,
    view2: torch.Tensor,
    step: int,
    max_steps: int,
    base_ema: float,
    base_lr: float,
    warmup_steps: int,
    device: torch.device,
) -> Dict[str, float]:
    """Execute one BYOL training step.

    Mirrors the _update_fn from the original repo's ByolExperiment class.

    Returns:
        dict of logged scalars
    """
    view1 = view1.to(device)
    view2 = view2.to(device)

    # -- Update learning rate --
    lr = cosine_lr_schedule(step, max_steps, base_lr, warmup_steps)
    for pg in optimizer.param_groups:
        pg['lr'] = lr

    # -- Forward: online network on both views --
    online.train()
    online_out1 = online(view1)  # {'projection': z, 'prediction': q}
    online_out2 = online(view2)

    # -- Forward: target network on both views (no grad) --
    with torch.no_grad():
        target.eval()
        target_out1 = target(view1)  # {'projection': z'}
        target_out2 = target(view2)

    # -- Symmetrized BYOL loss (paper Eq. 2, symmetrized) --
    # L = loss(q_θ(z_θ^v1), sg(z'_ξ^v2)) + loss(q_θ(z_θ^v2), sg(z'_ξ^v1))
    loss_12 = regression_loss(
        online_out1['prediction'],
        target_out2['projection'].detach(),
    ).mean()

    loss_21 = regression_loss(
        online_out2['prediction'],
        target_out1['projection'].detach(),
    ).mean()

    loss = loss_12 + loss_21

    # -- Backward + optimizer step (online only) --
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # -- EMA update of target network --
    tau = cosine_ema_schedule(step, base_ema, max_steps)
    update_target_ema(online, target, tau)

    return {
        'loss': loss.item(),
        'loss_12': loss_12.item(),
        'loss_21': loss_21.item(),
        'lr': lr,
        'tau': tau,
    }


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def pretrain(cfg: dict, device: Optional[torch.device] = None):
    """Run BYOL pretraining.

    Args:
        cfg: config dict from lightsheet_3d.get_config()
        device: torch device (auto-detected if None)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(cfg['random_seed'])
    print(f"Device: {device}")
    print(f"Config: {cfg['num_epochs']} epochs, batch_size={cfg['batch_size']}, "
          f"max_steps={cfg['max_steps']}")

    # -- Create networks --
    online, target = create_byol_pair(
        in_channels=cfg['encoder']['in_channels'],
        base_channels=cfg['encoder']['base_channels'],
        num_levels=cfg['encoder']['num_levels'],
        encoder_channels=cfg['encoder']['channels'],
        use_residuals=cfg['encoder']['use_residuals'],
        projector_hidden_dim=cfg['projector']['hidden_dim'],
        projector_output_dim=cfg['projector']['output_dim'],
        predictor_hidden_dim=cfg['predictor']['hidden_dim'],
    )
    online = online.to(device)
    target = target.to(device)

    n_online = sum(p.numel() for p in online.parameters())
    n_target = sum(p.numel() for p in target.parameters())
    print(f"Online: {n_online:,} params | Target: {n_target:,} params")

    # -- Optimizer: AdamW (not LARS -- batch size too small) --
    # Exclude bias and norm params from weight decay (paper convention)
    decay_params = []
    no_decay_params = []
    for name, p in online.named_parameters():
        if not p.requires_grad:
            continue
        if 'bias' in name or 'norm' in name or 'bn' in name:
            no_decay_params.append(p)
        else:
            decay_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': cfg['optimizer']['weight_decay']},
        {'params': no_decay_params, 'weight_decay': 0.0},
    ], lr=cfg['optimizer']['lr'], betas=tuple(cfg['optimizer']['betas']))

    # -- DataLoader --
    use_synthetic = cfg['data']['data_dir'] is None
    loader = create_byol_dataloader(
        data_dir=cfg['data']['data_dir'],
        batch_size=cfg['batch_size'],
        crop_size=cfg['data']['crop_size'],
        min_overlap=cfg['data']['min_overlap'],
        num_workers=cfg['data']['num_workers'],
        pin_memory=cfg['data']['pin_memory'],
        synthetic=use_synthetic,
        synthetic_num_samples=max(cfg['max_steps'] * cfg['batch_size'], 100),
        synthetic_volume_size=cfg['data'].get('volume_size', 32),
    )

    # -- Checkpoint dir --
    ckpt_dir = Path(cfg['checkpoint']['save_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # -- Training loop --
    global_step = 0
    log_interval = cfg['logging']['log_every_steps']
    save_interval = cfg['checkpoint']['save_every_epochs']

    print(f"\n{'='*60}")
    print(f"Starting BYOL pretraining")
    print(f"{'='*60}\n")

    for epoch in range(cfg['num_epochs']):
        epoch_loss = 0.0
        epoch_steps = 0
        t0 = time.time()

        for batch in loader:
            if global_step >= cfg['max_steps']:
                break

            logs = train_step(
                online=online,
                target=target,
                optimizer=optimizer,
                view1=batch['view1'],
                view2=batch['view2'],
                step=global_step,
                max_steps=cfg['max_steps'],
                base_ema=cfg['base_target_ema'],
                base_lr=cfg['optimizer']['lr'],
                warmup_steps=cfg['lr_schedule']['warmup_steps'],
                device=device,
            )

            epoch_loss += logs['loss']
            epoch_steps += 1
            global_step += 1

            if global_step % log_interval == 0:
                print(f"  step {global_step:>6d}/{cfg['max_steps']} | "
                      f"loss={logs['loss']:.4f} | "
                      f"lr={logs['lr']:.2e} | "
                      f"tau={logs['tau']:.6f}")

        if global_step >= cfg['max_steps']:
            break

        dt = time.time() - t0
        avg_loss = epoch_loss / max(epoch_steps, 1)
        print(f"Epoch {epoch+1:>4d}/{cfg['num_epochs']} | "
              f"avg_loss={avg_loss:.4f} | "
              f"{dt:.1f}s | "
              f"step={global_step}")

        # -- Save checkpoint --
        if (epoch + 1) % save_interval == 0 or (epoch + 1) == cfg['num_epochs']:
            ckpt_path = ckpt_dir / f"byol3d_epoch{epoch+1:04d}.pt"
            torch.save({
                'epoch': epoch + 1,
                'global_step': global_step,
                'online_state_dict': online.state_dict(),
                'target_state_dict': target.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg,
            }, ckpt_path)
            print(f"  -> Saved checkpoint: {ckpt_path}")

    # -- Save final --
    final_path = ckpt_dir / cfg['checkpoint']['filename']
    torch.save({
        'epoch': cfg['num_epochs'],
        'global_step': global_step,
        'online_state_dict': online.state_dict(),
        'target_state_dict': target.state_dict(),
        'encoder_state_dict': online.encoder.state_dict(),
        'config': cfg,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    print(f"Encoder state_dict key: 'encoder_state_dict' "
          f"(use for SegmentationUNet3D.from_byol_encoder)")

    return online, target


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='BYOL 3D Pretraining')
    parser.add_argument('--config', choices=['full', 'smoke'], default='smoke',
                        help='Config preset: "full" for cluster, "smoke" for local test')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Override checkpoint directory')
    args = parser.parse_args()

    if args.config == 'smoke':
        cfg = get_smoke_test_config()
    else:
        cfg = get_config()

    # Apply overrides
    if args.epochs is not None:
        cfg['num_epochs'] = args.epochs
        cfg['max_steps'] = args.epochs * cfg['steps_per_epoch']
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.data_dir is not None:
        cfg['data']['data_dir'] = args.data_dir
    if args.save_dir is not None:
        cfg['checkpoint']['save_dir'] = args.save_dir

    pretrain(cfg)
