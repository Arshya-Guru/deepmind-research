# ============================================================================
# FILE 2: byol3d/finetune_lightning.py
# Lightning-based segmentation fine-tuning script
# ============================================================================

"""
Segmentation fine-tuning with BYOL-pretrained encoder (Lightning).

Loads pretrained encoder, attaches U-Net decoder, trains with Dice+BCE loss.
Uses nnUNet splits for train/val, blosc2 data loading.

Usage:
    # Full fine-tuning with BYOL encoder
    python -m byol3d.finetune_lightning \
        --checkpoint ../byol_work/8/byol3d_pretrain.pt \
        --data-dir /nfs/khan/trainees/apooladi/abeta/nnssl/nnunet_data/preprocessed/Dataset001_ABetaPlaques \
        --epochs 200 --batch-size 2 --lr 5e-4 --precision bf16-mixed

    # Frozen encoder (linear probe)
    python -m byol3d.finetune_lightning \
        --checkpoint ../byol_work/8/byol3d_pretrain.pt \
        --data-dir ... --freeze-encoder --epochs 100

    # From scratch (no pretraining baseline)
    python -m byol3d.finetune_lightning \
        --data-dir ... --epochs 200
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from byol3d.utils.networks_3d import UNetEncoder3D, SegmentationUNet3D
from byol3d.utils.dataset_seg_blosc2 import (
    create_seg_dataloaders,
    _seg_worker_init_fn,
)


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------

def dice_loss(pred: torch.Tensor, target: torch.Tensor,
              smooth: float = 1.0) -> torch.Tensor:
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return 1.0 - dice.mean()


def dice_bce_loss(logits, target, dice_weight=0.5, bce_weight=0.5):
    probs = torch.sigmoid(logits)
    d = dice_loss(probs, target)
    b = F.binary_cross_entropy_with_logits(logits, target)
    return dice_weight * d + bce_weight * b


def compute_dice_score(logits: torch.Tensor, target: torch.Tensor,
                       threshold: float = 0.5) -> torch.Tensor:
    """Compute hard Dice score for monitoring."""
    pred = (torch.sigmoid(logits) > threshold).float()
    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    intersection = (pred_flat * target_flat).sum(1)
    union = pred_flat.sum(1) + target_flat.sum(1)
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    return dice.mean()


# ---------------------------------------------------------------------------
# Load pretrained encoder
# ---------------------------------------------------------------------------

def load_pretrained_encoder(checkpoint_path: str, device='cpu') -> UNetEncoder3D:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt['config']
    
    encoder = UNetEncoder3D(
        in_channels=cfg['encoder']['in_channels'],
        base_channels=cfg['encoder']['base_channels'],
        num_levels=cfg['encoder']['num_levels'],
        channels=cfg['encoder']['channels'],
        use_residuals=cfg['encoder']['use_residuals'],
    )
    
    if 'encoder_state_dict' in ckpt:
        encoder.load_state_dict(ckpt['encoder_state_dict'])
        print(f"Loaded encoder from: {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    else:
        online_sd = ckpt['online_state_dict']
        encoder_sd = {k.replace('encoder.', ''): v
                      for k, v in online_sd.items() if k.startswith('encoder.')}
        encoder.load_state_dict(encoder_sd)
        print(f"Loaded encoder (from online_state_dict): {checkpoint_path}")
    
    return encoder


# ---------------------------------------------------------------------------
# Lightning Module
# ---------------------------------------------------------------------------

class SegmentationLightningModule(pl.LightningModule):
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        freeze_encoder: bool = False,
        lr: float = 5e-4,
        encoder_lr_factor: float = 0.1,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 5,
        max_epochs: int = 200,
        steps_per_epoch: int = 100,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.encoder_lr_factor = encoder_lr_factor
        self.weight_decay = weight_decay
        self.freeze_encoder = freeze_encoder
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.steps_per_epoch = steps_per_epoch
        
        # Build model
        if checkpoint_path is not None:
            encoder = load_pretrained_encoder(checkpoint_path)
            self.model = SegmentationUNet3D.from_byol_encoder(
                encoder, num_classes=1, freeze_encoder=freeze_encoder)
            mode = "frozen" if freeze_encoder else "fine-tune"
            print(f"Model: pretrained encoder ({mode})")
        else:
            encoder = UNetEncoder3D(
                in_channels=1, base_channels=32, num_levels=5, use_residuals=True)
            self.model = SegmentationUNet3D(encoder=encoder, num_classes=1)
            print("Model: training from scratch (no pretraining)")
        
        n_total = sum(p.numel() for p in self.model.parameters())
        n_train = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  {n_total:,} total params, {n_train:,} trainable")
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']
        logits = self(images)
        
        loss = dice_bce_loss(logits, masks)
        dice = compute_dice_score(logits, masks)
        
        self.log('train/loss', loss, prog_bar=True, sync_dist=True)
        self.log('train/dice', dice, prog_bar=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        images = batch['image']
        masks = batch['mask']
        logits = self(images)
        
        loss = dice_bce_loss(logits, masks)
        dice = compute_dice_score(logits, masks)
        
        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/dice', dice, prog_bar=True, sync_dist=True)
        return {'val_loss': loss, 'val_dice': dice}
    
    def configure_optimizers(self):
        if self.freeze_encoder:
            optimizer = torch.optim.AdamW(
                self.model.decoder.parameters(),
                lr=self.lr, weight_decay=self.weight_decay)
        else:
            # Differential LR: encoder at 0.1x
            optimizer = torch.optim.AdamW([
                {'params': self.model.encoder.parameters(),
                 'lr': self.lr * self.encoder_lr_factor},
                {'params': self.model.decoder.parameters(),
                 'lr': self.lr},
            ], weight_decay=self.weight_decay)
        
        # Cosine annealing with warmup
        warmup_steps = self.warmup_epochs * self.steps_per_epoch
        max_steps = self.max_epochs * self.steps_per_epoch
        
        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
            return 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress))
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {'scheduler': scheduler, 'interval': 'step', 'frequency': 1},
        }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Segmentation Fine-tuning (Lightning)')
    
    warnings.filterwarnings('ignore', message='.*AccumulateGrad.*stream.*')
    
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='BYOL pretrain checkpoint (.pt). Omit to train from scratch.')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to Dataset001_ABetaPlaques/')
    parser.add_argument('--save-dir', type=str, default='./seg_work',
                        help='Output directory for checkpoints')
    parser.add_argument('--fold', type=int, default=0,
                        help='nnUNet split fold (default: 0)')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--encoder-lr-factor', type=float, default=0.1,
                        help='LR multiplier for encoder (default: 0.1)')
    parser.add_argument('--freeze-encoder', action='store_true',
                        help='Freeze encoder, only train decoder')
    parser.add_argument('--crop-size', type=int, default=96)
    parser.add_argument('--crops-per-volume', type=int, default=4,
                        help='Virtual epoch multiplier (random crops per volume)')
    parser.add_argument('--num-workers', type=int, default=8)
    parser.add_argument('--precision', type=str, default='32',
                        choices=['32', '16-mixed', 'bf16-mixed'])
    parser.add_argument('--devices', type=int, default=None)
    parser.add_argument('--strategy', type=str, default='auto')
    parser.add_argument('--val-every', type=int, default=10,
                        help='Validate every N epochs')
    
    args = parser.parse_args()
    
    # -- Create dataloaders --
    train_loader, val_loader = create_seg_dataloaders(
        data_dir=args.data_dir,
        fold=args.fold,
        crop_size=args.crop_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        crops_per_volume=args.crops_per_volume,
    )
    
    steps_per_epoch = len(train_loader)
    print(f"Steps per epoch: {steps_per_epoch}")
    
    # -- Build Lightning module --
    module = SegmentationLightningModule(
        checkpoint_path=args.checkpoint,
        freeze_encoder=args.freeze_encoder,
        lr=args.lr,
        encoder_lr_factor=args.encoder_lr_factor,
        max_epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
    )
    
    # -- Callbacks --
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    ckpt_callback = ModelCheckpoint(
        dirpath=str(save_dir),
        filename='seg-{epoch:04d}-{val/dice:.4f}',
        monitor='val/dice',
        mode='max',
        save_top_k=3,
        save_last=True,
        every_n_epochs=args.val_every,
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    # -- Trainer --
    devices = args.devices or 'auto'
    
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator='auto',
        devices=devices,
        strategy=args.strategy,
        precision=args.precision,
        callbacks=[ckpt_callback, lr_monitor],
        check_val_every_n_epoch=args.val_every,
        log_every_n_steps=5,
        default_root_dir=str(save_dir),
        enable_progress_bar=True,
    )
    
    # -- Train --
    trainer.fit(module, train_loader, val_loader)
    
    # -- Save final encoder + full model for inference --
    final_path = save_dir / 'seg_final.pt'
    torch.save({
        'model_state_dict': module.model.state_dict(),
        'encoder_state_dict': module.model.encoder.state_dict(),
        'hparams': dict(module.hparams),
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    print(f"Best val/dice: {ckpt_callback.best_model_score}")


if __name__ == '__main__':
    main()
