"""
Segmentation fine-tuning with BYOL-pretrained encoder.

Phase 2 of the pipeline: load pretrained encoder weights, attach a
fresh 3D U-Net decoder, and train on labeled Abeta masks.

Loss: Dice + BCE (standard for medical image segmentation).

Usage:
    python -m byol.finetune_segmentation \
        --checkpoint /path/to/byol3d_pretrain.pt \
        --data-dir /path/to/labeled_data \
        --freeze-encoder   # optional: linear probe mode
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from byol3d.utils.networks_3d import (
    UNetEncoder3D,
    SegmentationUNet3D,
)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    """Soft Dice loss for binary segmentation.

    Args:
        pred: [B, 1, D, H, W] sigmoid probabilities
        target: [B, 1, D, H, W] binary masks
        smooth: smoothing factor to avoid division by zero
    """
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)

    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)

    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def dice_bce_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    dice_weight: float = 0.5,
    bce_weight: float = 0.5,
) -> torch.Tensor:
    """Combined Dice + BCE loss.

    Standard for medical image segmentation (nnU-Net uses this).
    """
    probs = torch.sigmoid(logits)
    d_loss = dice_loss(probs, target)
    b_loss = F.binary_cross_entropy_with_logits(logits, target)
    return dice_weight * d_loss + bce_weight * b_loss


# ---------------------------------------------------------------------------
# Load pretrained encoder
# ---------------------------------------------------------------------------

def load_pretrained_encoder(
    checkpoint_path: str,
    device: torch.device,
) -> UNetEncoder3D:
    """Load encoder weights from a BYOL pretraining checkpoint.

    The checkpoint contains 'encoder_state_dict' saved by byol_pretrain_3d.py.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct encoder from saved config
    cfg = ckpt['config']
    encoder = UNetEncoder3D(
        in_channels=cfg['encoder']['in_channels'],
        base_channels=cfg['encoder']['base_channels'],
        num_levels=cfg['encoder']['num_levels'],
        channels=cfg['encoder']['channels'],
        use_residuals=cfg['encoder']['use_residuals'],
    )

    # Load weights
    if 'encoder_state_dict' in ckpt:
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        print(f"Loaded encoder weights from: {checkpoint_path}")
    else:
        # Fallback: extract encoder keys from online_state_dict
        online_sd = ckpt['online_state_dict']
        encoder_sd = {k.replace('encoder.', ''): v
                      for k, v in online_sd.items()
                      if k.startswith('encoder.')}
        encoder.load_state_dict(encoder_sd)
        print(f"Loaded encoder weights (extracted from online) from: {checkpoint_path}")

    return encoder


# ---------------------------------------------------------------------------
# Fine-tuning loop (skeleton — you'll adapt to your data loading)
# ---------------------------------------------------------------------------

def finetune(
    checkpoint_path: Optional[str] = None,
    data_dir: Optional[str] = None,
    freeze_encoder: bool = False,
    num_epochs: int = 100,
    batch_size: int = 2,
    lr: float = 1e-3,
    encoder_lr_factor: float = 0.1,
    device: Optional[torch.device] = None,
):
    """Fine-tune segmentation model.

    Args:
        checkpoint_path: path to BYOL pretraining checkpoint, or None to
            train from scratch.
        data_dir: path to labeled data directory.
        freeze_encoder: if True, only train the decoder (linear probe).
        num_epochs: number of fine-tuning epochs.
        batch_size: per-GPU batch size.
        lr: base learning rate.
        encoder_lr_factor: LR multiplier for encoder (lower = more frozen).
            Only used when freeze_encoder=False.
        device: torch device.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # -- Build model --
    if checkpoint_path is not None:
        encoder = load_pretrained_encoder(checkpoint_path, device)
        model = SegmentationUNet3D.from_byol_encoder(
            encoder, num_classes=1, freeze_encoder=freeze_encoder)
        print(f"Loaded pretrained encoder, freeze={freeze_encoder}")
    else:
        encoder = UNetEncoder3D(in_channels=1, base_channels=32, num_levels=5)
        model = SegmentationUNet3D(encoder=encoder, num_classes=1)
        print("Training from scratch (no pretrained encoder)")

    model = model.to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {n_total:,} total params, {n_train:,} trainable")

    # -- Optimizer with differential LR --
    if freeze_encoder:
        optimizer = torch.optim.AdamW(
            model.decoder.parameters(), lr=lr, weight_decay=1e-4)
    else:
        optimizer = torch.optim.AdamW([
            {'params': model.encoder.parameters(), 'lr': lr * encoder_lr_factor},
            {'params': model.decoder.parameters(), 'lr': lr},
        ], weight_decay=1e-4)

    # -- Training loop (placeholder — integrate with your data loader) --
    print(f"\nFine-tuning for {num_epochs} epochs...")
    print("NOTE: Replace the synthetic data loop below with your labeled DataLoader\n")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        # ---- REPLACE THIS with your real labeled DataLoader ----
        for step in range(5):  # placeholder: 5 fake batches per epoch
            images = torch.rand(batch_size, 1, 96, 96, 96, device=device)
            masks = (torch.rand(batch_size, 1, 96, 96, 96, device=device) > 0.8).float()

            logits = model(images)
            loss = dice_bce_loss(logits, masks)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        # ---- END PLACEHOLDER ----

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  Epoch {epoch+1:>4d}/{num_epochs} | loss={epoch_loss/5:.4f}")

    print("\nFine-tuning complete.")
    return model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Segmentation Fine-tuning')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to BYOL pretraining checkpoint')
    parser.add_argument('--data-dir', type=str, default=None,
                        help='Path to labeled data directory')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze encoder (linear probe mode)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()

    finetune(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        freeze_encoder=args.freeze_encoder,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
    )
