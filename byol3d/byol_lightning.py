"""
BYOL 3D Lightning module for multi-GPU pretraining.

Wraps our existing networks_3d.py / augmentations_3d.py / dataset_blosc2.py
in a PyTorch Lightning module for automatic DDP multi-GPU support.

Critical BYOL-specific DDP considerations (collapse prevention):
  1. Target network is NOT optimized — its params are frozen (requires_grad=False)
     so DDP's allreduce ignores them.
  2. EMA updates happen after each training step. Since DDP guarantees identical
     online params across all ranks after allreduce, each rank does EMA independently
     and stays in sync.
  3. SyncBatchNorm is optional but recommended for the MLP heads' BatchNorm1d
     when per-GPU batch size is small (2-4). The encoder uses InstanceNorm3d
     which is batch-size independent.
  4. The target network is a plain attribute — Lightning wraps the full module in
     DDP, but frozen params don't participate in gradient communication.

Usage:
    # Single GPU (backward compatible):
    python -m byol3d.byol_lightning --config full --devices 1

    # Multi-GPU DDP:
    python -m byol3d.byol_lightning --config full --devices 4

    # Smoke test:
    python -m byol3d.byol_lightning --config smoke

    # Resume from checkpoint:
    python -m byol3d.byol_lightning --config full --resume /path/to/byol3d_epoch0050.pt
"""

from __future__ import annotations

import argparse
import copy
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
    from lightning.pytorch.strategies import DDPStrategy
except ImportError:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
    from pytorch_lightning.strategies import DDPStrategy

from byol3d.configs.lightsheet_3d import get_config, get_smoke_test_config
from byol3d.utils.networks_3d import (
    create_byol_pair,
    regression_loss,
    update_target_ema,
    cosine_ema_schedule,
)
from byol3d.utils.dataset_blosc2 import create_byol_dataloader


# ---------------------------------------------------------------------------
# Custom LR scheduler (cosine with warmup) for Lightning
# ---------------------------------------------------------------------------

class CosineWarmupScheduler(torch.optim.lr_scheduler.LambdaLR):
    """Cosine LR schedule with linear warmup, matching our existing schedule."""

    def __init__(self, optimizer, warmup_steps: int, max_steps: int):
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps

        def lr_lambda(step):
            if warmup_steps > 0 and step < warmup_steps:
                return step / warmup_steps
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            progress = min(progress, 1.0)
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        super().__init__(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# BYOL Lightning Module
# ---------------------------------------------------------------------------

class BYOLLightningModule(pl.LightningModule):
    """BYOL pretraining wrapped in PyTorch Lightning.

    This module:
      - Creates online + target networks from config
      - Freezes target params (no DDP gradient sync overhead)
      - Runs symmetrized BYOL loss in training_step
      - Updates target via EMA in on_train_batch_end
      - Optionally applies SyncBatchNorm to MLP heads
    """

    def __init__(self, cfg: Dict[str, Any]):
        super().__init__()
        self.cfg = cfg
        # save_hyperparameters stores cfg in self.hparams for checkpoint reload
        self.save_hyperparameters(cfg)

        # -- Build online + target networks --
        self.online, self.target = create_byol_pair(
            in_channels=cfg['encoder']['in_channels'],
            base_channels=cfg['encoder']['base_channels'],
            num_levels=cfg['encoder']['num_levels'],
            encoder_channels=cfg['encoder']['channels'],
            use_residuals=cfg['encoder']['use_residuals'],
            projector_hidden_dim=cfg['projector']['hidden_dim'],
            projector_output_dim=cfg['projector']['output_dim'],
            predictor_hidden_dim=cfg['predictor']['hidden_dim'],
        )

        # -- Freeze target network --
        # This ensures DDP does not include target params in allreduce.
        for p in self.target.parameters():
            p.requires_grad = False

        # Store schedule params
        self.base_ema = cfg['base_target_ema']
        self.max_steps = cfg['max_steps']

        # Automatic optimization is ON (Lightning handles backward + step)
        self.automatic_optimization = True

    def forward(self, x):
        """Forward through online encoder only (for inference/export)."""
        return self.online.encoder(x)

    # ----- Training step -----

    def training_step(self, batch, batch_idx):
        view1 = batch['view1']
        view2 = batch['view2']

        # Online forward (with grad, DDP handles allreduce)
        self.online.train()
        online_out1 = self.online(view1)
        online_out2 = self.online(view2)

        # Target forward (no grad, frozen)
        with torch.no_grad():
            self.target.eval()
            target_out1 = self.target(view1)
            target_out2 = self.target(view2)

        # Symmetrized BYOL loss (paper Eq. 2)
        loss_12 = regression_loss(
            online_out1['prediction'],
            target_out2['projection'].detach(),
        ).mean()

        loss_21 = regression_loss(
            online_out2['prediction'],
            target_out1['projection'].detach(),
        ).mean()

        loss = loss_12 + loss_21

        # -- Compute current tau for logging --
        tau = cosine_ema_schedule(
            self.global_step, self.base_ema, self.max_steps
        )

        # Log metrics
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/loss_12', loss_12, sync_dist=True)
        self.log('train/loss_21', loss_21, sync_dist=True)
        self.log('train/tau', tau, prog_bar=True)
        self.log('train/lr', self.optimizers().param_groups[0]['lr'], prog_bar=True)

        return loss

    # ----- EMA update (after each training step) -----

    def on_train_batch_end(self, outputs, batch, batch_idx):
        """Update target network via EMA after each step.

        This runs AFTER Lightning's backward + optimizer step + allreduce,
        so all ranks have identical online params → identical EMA updates
        → target stays synchronized across GPUs without broadcasting.
        """
        tau = cosine_ema_schedule(
            self.global_step, self.base_ema, self.max_steps
        )
        # EMA from the unwrapped online network
        # (self.online is the real module, Lightning/DDP wraps self at a higher level)
        update_target_ema(self.online, self.target, tau)

    # ----- Optimizer + LR scheduler -----

    def configure_optimizers(self):
        # Separate weight-decay and no-weight-decay params (online only)
        # Target params are frozen, so they're excluded automatically
        decay_params = []
        no_decay_params = []
        for name, p in self.online.named_parameters():
            if not p.requires_grad:
                continue
            if 'bias' in name or 'norm' in name:
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        optimizer = torch.optim.AdamW([
            {'params': decay_params,
             'weight_decay': self.cfg['optimizer']['weight_decay']},
            {'params': no_decay_params,
             'weight_decay': 0.0},
        ],
            lr=self.cfg['optimizer']['lr'],
            betas=tuple(self.cfg['optimizer']['betas']),
        )

        scheduler = CosineWarmupScheduler(
            optimizer,
            warmup_steps=self.cfg['lr_schedule']['warmup_steps'],
            max_steps=self.max_steps,
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',   # update LR every step, not every epoch
                'frequency': 1,
            },
        }

    # ----- DataLoader -----

    def train_dataloader(self):
        use_synthetic = self.cfg['data']['data_dir'] is None
        return create_byol_dataloader(
            data_dir=self.cfg['data']['data_dir'],
            batch_size=self.cfg['batch_size'],
            crop_size=self.cfg['data']['crop_size'],
            min_overlap=self.cfg['data']['min_overlap'],
            num_workers=self.cfg['data']['num_workers'],
            pin_memory=self.cfg['data']['pin_memory'],
            synthetic=use_synthetic,
            synthetic_num_samples=max(
                self.cfg['max_steps'] * self.cfg['batch_size'], 100),
            synthetic_volume_size=self.cfg['data'].get('volume_size', 32),
            cache_dir=self.cfg['data'].get('cache_dir', None),
        )

    # ----- Resume from BYOL checkpoint -----

    def load_byol_checkpoint(self, ckpt_path: str):
        """Load model weights (and optionally optimizer/scheduler) from a
        BYOL checkpoint saved by BYOLCheckpointCallback.

        This restores online + target state dicts. If the checkpoint also
        contains optimizer/scheduler state (saved by updated callback),
        those will be restored after configure_optimizers() runs via
        the resume_optimizer_state stored on the module.

        Args:
            ckpt_path: path to byol3d_epoch*.pt checkpoint file.

        Returns:
            dict with 'epoch' and 'global_step' from the checkpoint.
        """
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

        # Restore model weights
        self.online.load_state_dict(ckpt['online_state_dict'])
        self.target.load_state_dict(ckpt['target_state_dict'])

        # Stash optimizer/scheduler state for restoration after
        # configure_optimizers() creates the optimizer (see main() below)
        if 'optimizer_state_dict' in ckpt:
            self._resume_optimizer_state = ckpt['optimizer_state_dict']
        if 'scheduler_state_dict' in ckpt:
            self._resume_scheduler_state = ckpt['scheduler_state_dict']

        resume_info = {
            'epoch': ckpt['epoch'],
            'global_step': ckpt['global_step'],
        }
        print(f'Loaded BYOL checkpoint: {ckpt_path}')
        print(f'  Completed epochs: {resume_info["epoch"]}, '
              f'global_step: {resume_info["global_step"]}')
        return resume_info


# ---------------------------------------------------------------------------
# Custom checkpoint callback (saves encoder_state_dict for fine-tuning)
# ---------------------------------------------------------------------------

class BYOLCheckpointCallback(pl.Callback):
    """Save encoder-only checkpoint compatible with SegmentationUNet3D.from_byol_encoder.

    Now also saves optimizer and LR scheduler state for resumable training.
    """

    def __init__(self, save_dir: str, save_every_epochs: int = 50):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.save_every_epochs = save_every_epochs

    def _get_optimizer_scheduler_state(self, trainer, pl_module):
        """Extract optimizer and scheduler state dicts."""
        opt_state = None
        sched_state = None
        if trainer.optimizers:
            opt_state = trainer.optimizers[0].state_dict()
        if trainer.lr_scheduler_configs:
            sched_state = trainer.lr_scheduler_configs[0].scheduler.state_dict()
        return opt_state, sched_state

    def on_train_epoch_end(self, trainer, pl_module):
        epoch = trainer.current_epoch + 1
        if epoch % self.save_every_epochs == 0 or epoch == trainer.max_epochs:
            ckpt_path = self.save_dir / f'byol3d_epoch{epoch:04d}.pt'
            opt_state, sched_state = self._get_optimizer_scheduler_state(
                trainer, pl_module)

            # Save in our standard format (compatible with finetune_segmentation.py)
            # Now also includes optimizer + scheduler for resumable training
            save_dict = {
                'epoch': epoch,
                'global_step': trainer.global_step,
                'online_state_dict': pl_module.online.state_dict(),
                'target_state_dict': pl_module.target.state_dict(),
                'encoder_state_dict': pl_module.online.encoder.state_dict(),
                'config': pl_module.cfg,
            }
            if opt_state is not None:
                save_dict['optimizer_state_dict'] = opt_state
            if sched_state is not None:
                save_dict['scheduler_state_dict'] = sched_state

            torch.save(save_dict, ckpt_path)
            if trainer.is_global_zero:
                print(f'  -> Saved BYOL checkpoint: {ckpt_path}')

    def on_train_end(self, trainer, pl_module):
        """Always save a final checkpoint."""
        final_path = self.save_dir / 'byol3d_pretrain.pt'
        opt_state, sched_state = self._get_optimizer_scheduler_state(
            trainer, pl_module)

        save_dict = {
            'epoch': trainer.current_epoch + 1,
            'global_step': trainer.global_step,
            'online_state_dict': pl_module.online.state_dict(),
            'target_state_dict': pl_module.target.state_dict(),
            'encoder_state_dict': pl_module.online.encoder.state_dict(),
            'config': pl_module.cfg,
        }
        if opt_state is not None:
            save_dict['optimizer_state_dict'] = opt_state
        if sched_state is not None:
            save_dict['scheduler_state_dict'] = sched_state

        torch.save(save_dict, final_path)
        if trainer.is_global_zero:
            print(f'Final model saved: {final_path}')
            print(f"Encoder state_dict key: 'encoder_state_dict' "
                  f"(use for SegmentationUNet3D.from_byol_encoder)")


# ---------------------------------------------------------------------------
# Resume helper: restore optimizer/scheduler + Lightning fit_loop counters
# ---------------------------------------------------------------------------

class ResumeCallback(pl.Callback):
    """Restores optimizer/scheduler state and fast-forwards Lightning's
    fit_loop counters so training resumes from the correct epoch/step.

    This is needed because we use a custom checkpoint format (not Lightning's
    native .ckpt), so we handle resume manually.
    """

    def __init__(self, resume_epoch: int, resume_global_step: int):
        self.resume_epoch = resume_epoch
        self.resume_global_step = resume_global_step

    def on_train_start(self, trainer, pl_module):
        """Restore optimizer/scheduler state after Lightning has created them."""
        # Restore optimizer state if available
        if hasattr(pl_module, '_resume_optimizer_state'):
            opt = trainer.optimizers[0]
            opt.load_state_dict(pl_module._resume_optimizer_state)
            del pl_module._resume_optimizer_state
            print(f'  Restored optimizer state (AdamW momentum buffers)')

        # Restore scheduler state if available
        if hasattr(pl_module, '_resume_scheduler_state'):
            sched = trainer.lr_scheduler_configs[0].scheduler
            sched.load_state_dict(pl_module._resume_scheduler_state)
            del pl_module._resume_scheduler_state
            print(f'  Restored LR scheduler state')
        elif self.resume_global_step > 0:
            # No scheduler state saved (old checkpoint format) — manually
            # fast-forward the LR scheduler to the correct step
            sched = trainer.lr_scheduler_configs[0].scheduler
            print(f'  Fast-forwarding LR scheduler by {self.resume_global_step} steps...')
            for _ in range(self.resume_global_step):
                sched.step()
            print(f'  LR scheduler at step {sched.last_epoch}, '
                  f'lr={trainer.optimizers[0].param_groups[0]["lr"]:.2e}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='BYOL 3D Lightning Trainer')

    # Suppress harmless DDP stream mismatch warning
    import warnings
    warnings.filterwarnings('ignore', message='.*AccumulateGrad.*stream.*')
    try:
        torch.autograd.graph.set_warn_on_accumulate_grad_stream_mismatch(False)
    except AttributeError:
        pass  # older PyTorch versions don't have this

    parser.add_argument('--config', type=str, default='full',
                        choices=['full', 'smoke'],
                        help='Config preset (default: full)')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override num_epochs from config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override per-GPU batch_size')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Override data directory')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Override checkpoint save directory')
    parser.add_argument('--devices', type=int, default=None,
                        help='Number of GPUs (default: auto-detect)')
    parser.add_argument('--strategy', type=str, default='auto',
                        help='Lightning strategy (auto, ddp, ddp_find_unused_parameters_true)')
    parser.add_argument('--sync-batchnorm', action='store_true',
                        help='Convert BatchNorm1d in MLP heads to SyncBatchNorm')
    parser.add_argument('--precision', type=str, default='32',
                        choices=['32', '16-mixed', 'bf16-mixed'],
                        help='Training precision (default: 32)')
    parser.add_argument('--num-samples', type=int, default=None,
                        help='Override dataset size (auto-detected from data-dir)')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Override num_workers for DataLoader')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='Local SSD cache dir (e.g. /tmp/byol_cache). '
                             'Warning: 10k patches need ~640GB. '
                             'Default: None (rely on OS page cache with sufficient RAM)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to BYOL checkpoint to resume training from '
                             '(e.g. /path/to/byol3d_epoch0050.pt)')
    args = parser.parse_args()

    # -- Build config --
    if args.config == 'smoke':
        cfg = get_smoke_test_config()
    else:
        cfg = get_config()

    # -- Apply CLI overrides --
    if args.epochs is not None:
        cfg['num_epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['batch_size'] = args.batch_size
    if args.data_dir is not None:
        cfg['data']['data_dir'] = args.data_dir
    if args.save_dir is not None:
        cfg['checkpoint']['save_dir'] = args.save_dir

    # -- Auto-detect dataset size for accurate max_steps --
    # With DDP, each GPU sees dataset_size // num_devices samples per epoch.
    # We count files once here so the progress bar and LR schedule are correct.
    data_dir = cfg['data'].get('data_dir')
    if data_dir is not None and args.num_samples is None:
        from pathlib import Path
        n_files = len(list(Path(data_dir).rglob('*.b2nd')))
        if n_files == 0:
            n_files = len(list(Path(data_dir).rglob('*.npy')))
        if n_files > 0:
            cfg['num_samples'] = n_files
            print(f'Auto-detected {n_files} files in {data_dir}')
    if args.num_samples is not None:
        cfg['num_samples'] = args.num_samples

    # Recompute steps with actual sample count and device count
    num_devices_for_calc = args.devices or (
        torch.cuda.device_count() if torch.cuda.is_available() else 1)
    samples_per_gpu = cfg.get('num_samples', 1000) // num_devices_for_calc
    cfg['steps_per_epoch'] = max(samples_per_gpu // cfg['batch_size'], 1)
    cfg['max_steps'] = cfg['num_epochs'] * cfg['steps_per_epoch']

    # Warmup: use warmup_epochs from config, but cap at 10% of total training
    # for short runs (e.g. --epochs 1 shouldn't spend the whole run warming up)
    warmup_epochs = cfg['lr_schedule'].get('warmup_epochs', 10)
    warmup_from_config = warmup_epochs * cfg['steps_per_epoch']
    warmup_cap = max(cfg['max_steps'] // 10, 1)  # at most 10% of total
    cfg['lr_schedule']['warmup_steps'] = min(warmup_from_config, warmup_cap)

    # -- Auto-detect devices --
    if args.devices is not None:
        num_devices = args.devices
    else:
        num_devices = torch.cuda.device_count() if torch.cuda.is_available() else 1

    # Override num_workers if specified
    if args.num_workers is not None:
        cfg['data']['num_workers'] = args.num_workers

    # Override cache_dir if specified
    if args.cache_dir is not None:
        cfg['data']['cache_dir'] = args.cache_dir

    accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'

    # Use tensor cores on A100/L40S for ~2x matmul speedup
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')

    # -- Strategy --
    if num_devices > 1:
        strategy = DDPStrategy(
            find_unused_parameters=False,
            # Static graph: all params are used every step → faster
            static_graph=True,
        )
    else:
        strategy = 'auto'

    # Override strategy if specified
    if args.strategy != 'auto':
        strategy = args.strategy

    # -- Create model --
    model = BYOLLightningModule(cfg)

    # -- Resume from checkpoint if specified --
    resume_epoch = 0
    resume_global_step = 0
    if args.resume is not None:
        resume_info = model.load_byol_checkpoint(args.resume)
        resume_epoch = resume_info['epoch']
        resume_global_step = resume_info['global_step']
        print(f'Will resume training from epoch {resume_epoch}, '
              f'step {resume_global_step}')

    # -- Optional SyncBatchNorm for MLP heads --
    if args.sync_batchnorm and num_devices > 1:
        # Only convert the online network's BatchNorm1d (in projector/predictor)
        # The encoder uses InstanceNorm3d which doesn't need syncing
        model.online.projector = nn.SyncBatchNorm.convert_model(model.online.projector)
        model.online.predictor = nn.SyncBatchNorm.convert_model(model.online.predictor)
        # Also sync target's projector BN (it will be overwritten by EMA,
        # but needs matching architecture for state_dict compatibility)
        model.target.projector = nn.SyncBatchNorm.convert_model(model.target.projector)
        print('SyncBatchNorm enabled for projector/predictor MLP heads')

    # -- Print info --
    n_online = sum(p.numel() for p in model.online.parameters())
    n_target = sum(p.numel() for p in model.target.parameters())
    print(f'Config: {cfg["num_epochs"]} epochs, batch_size={cfg["batch_size"]} per GPU, '
          f'max_steps={cfg["max_steps"]}')
    print(f'  steps_per_epoch={cfg["steps_per_epoch"]}, '
          f'warmup_steps={cfg["lr_schedule"]["warmup_steps"]}')
    print(f'Online: {n_online:,} params | Target: {n_target:,} params')
    print(f'Devices: {num_devices} x {accelerator}')
    if num_devices > 1:
        print(f'Effective batch size: {cfg["batch_size"] * num_devices}')
    print(f'Data workers: {cfg["data"]["num_workers"]} per process')
    if cfg['data'].get('cache_dir'):
        print(f'Cache dir: {cfg["data"]["cache_dir"]} '
              f'(first epoch slow, subsequent fast)')

    # -- Callbacks --
    callbacks = [
        BYOLCheckpointCallback(
            save_dir=cfg['checkpoint']['save_dir'],
            save_every_epochs=cfg['checkpoint']['save_every_epochs'],
        ),
    ]

    # Add resume callback if resuming
    if args.resume is not None:
        callbacks.append(ResumeCallback(
            resume_epoch=resume_epoch,
            resume_global_step=resume_global_step,
        ))

    # -- Trainer --
    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=num_devices,
        strategy=strategy,
        max_epochs=cfg['num_epochs'],
        max_steps=cfg['max_steps'],
        callbacks=callbacks,
        enable_progress_bar=True,
        log_every_n_steps=cfg['logging']['log_every_steps'],
        precision=args.precision,
        # Gradient clipping (conservative, prevents instability)
        gradient_clip_val=1.0,
        gradient_clip_algorithm='norm',
        # No validation during pretraining
        enable_checkpointing=False,  # we use our own callback
        num_sanity_val_steps=0,
    )

    # -- Fast-forward fit_loop if resuming --
    if args.resume is not None and resume_epoch > 0:
        # Tell Lightning to start from the correct epoch and step.
        # This skips already-completed epochs rather than re-running them.
        trainer.fit_loop.epoch_progress.current.completed = resume_epoch
        trainer.fit_loop.epoch_progress.current.processed = resume_epoch
        print(f'Fast-forwarded Lightning fit_loop to epoch {resume_epoch}')

    # -- Train! --
    trainer.fit(model)


if __name__ == '__main__':
    main()