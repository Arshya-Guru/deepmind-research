# byol3d

BYOL (Bootstrap Your Own Latent) self-supervised pretraining adapted for 3D lightsheet microscopy, targeting Aβ plaque segmentation.

Reimplemented in PyTorch from [DeepMind's original JAX/Haiku codebase](https://github.com/deepmind/deepmind-research/tree/master/byol) (Grill et al., 2020). The original 2D ResNet + ImageNet pipeline is replaced with a 3D U-Net encoder operating on single-channel fluorescence volumes.

## Why BYOL over SimCLR

- **No negative pairs** → robust at small batch sizes (2–16). SimCLR degrades catastrophically below batch 256.
- **Augmentation-tolerant** → dropping augmentations costs ~3pt for BYOL vs ~10pt for SimCLR (paper Table 16). Critical for single-channel microscopy where color jittering/solarization don't apply.
- **EMA target network** → more stable training signal than contrastive methods.

## Quick start

```bash
# Install
pixi install

# Smoke test (CPU, ~5 sec)
pixi run smoke

# Unit tests
pixi run test
```

## Pretraining

### CLI reference

```
python -m byol3d.byol_lightning --config {full|smoke} [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | required | `full` for real training, `smoke` for CPU test |
| `--epochs` | 300 | Number of training epochs |
| `--batch-size` | 8 | Per-GPU batch size |
| `--data-dir` | config default | Path to preprocessed `.b2nd` patches |
| `--save-dir` | config default | Checkpoint output directory |
| `--num-workers` | 8 | DataLoader workers per process |
| `--precision` | `32` | `32`, `16-mixed`, or `bf16-mixed` |
| `--devices` | auto | Number of GPUs (auto-detected) |
| `--strategy` | `auto` | `auto`, `ddp`, `ddp_find_unused_parameters_true` |
| `--sync-batchnorm` | off | Convert MLP head BatchNorm to SyncBatchNorm (multi-GPU) |

### Example: single GPU, batch size 16

Uses one GPU with per-GPU batch 16 → effective batch size 16. Good for quick iteration or smaller GPU memory (e.g. single A100-40GB).

```bash
salloc --time=2-00:00 --cpus-per-task=16 --mem=128000 --gres=gpu:1

pixi shell
python -m byol3d.byol_lightning --config full \
    --epochs 300 --batch-size 16 \
    --data-dir /nfs/khan/trainees/apooladi/abeta/nnssl_data/8/nnssl_data/preprocessed \
    --save-dir ../byol_work/8/ \
    --num-workers 8 --precision bf16-mixed --devices 1
```

**What this does:**
- 10,000 files / batch 16 = **625 steps per epoch**
- bf16-mixed halves VRAM usage → batch 16 fits comfortably on one A100
- 8 workers feed data from NFS, OS page cache makes epoch 2+ ~5x faster
- Epoch 1: ~1hr (cold NFS reads). Epochs 2+: ~10-15 min each (page cache warm)
- **~3 days total** for 300 epochs

### Example: 2 GPUs, batch size 64

Uses both A100s with per-GPU batch 32 → effective batch size 64. Lightning auto-detects 2 GPUs and uses DDP (DistributedDataParallel). Gradients are allreduced across GPUs so the optimizer sees the full batch.

```bash
salloc --time=2-00:00 --cpus-per-task=32 --mem=256000 --gres=gpu:2

pixi shell
python -m byol3d.byol_lightning --config full \
    --epochs 300 --batch-size 32 \
    --data-dir /nfs/khan/trainees/apooladi/abeta/nnssl_data/8/nnssl_data/preprocessed \
    --save-dir ../byol_work/8/ \
    --num-workers 16 --precision bf16-mixed
```

**What this does:**
- Lightning auto-detects 2 GPUs → DDP with DistributedSampler
- Each GPU processes 32 samples, effective batch = 32 × 2 = **64**
- 10,000 files / 64 = **~156 steps per epoch** (much faster than single-GPU)
- 16 workers per process (32 total across both ranks)
- DistributedSampler splits files: each GPU sees 5,000 unique files per epoch
- BatchNorm in projector/predictor computes stats per-GPU (32 samples each — plenty for stable BN)
- **~1-1.5 days total** for 300 epochs

### VRAM estimates (per GPU)

| Per-GPU Batch | Precision | Approx. VRAM | Fits on |
|---------------|-----------|--------------|---------|
| 8 | bf16-mixed | ~18 GB | A100-40GB ✓ |
| 16 | bf16-mixed | ~28 GB | A100-40GB ✓ |
| 32 | bf16-mixed | ~38 GB | A100-40GB (tight), A100-80GB ✓ |
| 64 | bf16-mixed | ~60 GB | A100-80GB only |

If batch 32 OOMs on A100-40GB, drop to `--batch-size 24` or `--batch-size 16`.

### Pixi tasks

| Task | What it does |
|------|--------------|
| `pixi run smoke` | Smoke test: synthetic data, no GPU, ~5 sec |
| `pixi run test` | Network unit tests (shapes, gradients, EMA, loss) |
| `pixi run pretrain` | Full pretraining: batch 16, 300 epochs |
| `pixi run pretrain-short` | Shorter pretraining: batch 16, 100 epochs |

To bypass pixi tasks and pass your own flags:

```bash
pixi run -- python -m byol3d.byol_lightning --config full --batch-size 32 --epochs 500
```

## Fine-tuning

```
python -m byol3d.finetune_segmentation [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | None | Path to BYOL `.pt` checkpoint. Omit to train from scratch. |
| `--data-dir` | None | Path to labeled segmentation data |
| `--freeze-encoder` | off | Freeze encoder, train decoder only (linear probe) |
| `--epochs` | 100 | Fine-tuning epochs |
| `--batch-size` | 2 | Per-GPU batch size |
| `--lr` | 1e-3 | Base learning rate |

**Examples:**

```bash
# Full fine-tuning with pretrained encoder
python -m byol3d.finetune_segmentation \
    --checkpoint ../byol_work/8/byol3d_pretrain.pt \
    --epochs 200 --lr 5e-4

# Linear probe: freeze encoder, only train decoder
python -m byol3d.finetune_segmentation \
    --checkpoint ../byol_work/8/byol3d_pretrain.pt \
    --freeze-encoder --epochs 100

# Train from scratch (baseline comparison — no pretraining)
python -m byol3d.finetune_segmentation --epochs 200 --batch-size 2
```

When `--freeze-encoder` is off (default), the encoder trains at 0.1× the base LR while the decoder trains at full LR.

## Config

All hyperparameters live in `byol3d/configs/lightsheet_3d.py`. CLI flags override values at runtime.

To change things without CLI flags (encoder depth, projector dims, augmentation params), edit the config directly:

```python
# byol3d/configs/lightsheet_3d.py

encoder=dict(
    in_channels=1,
    base_channels=32,
    num_levels=4,        # was 5 — fewer levels = smaller model
),

projector=dict(
    hidden_dim=1024,     # was 2048
    output_dim=128,      # was 256
),
```

Augmentation parameters are in `byol3d/utils/augmentations_3d.py` in the `augment_config` dict. The asymmetry between view1 and view2 (blur probability 0.5 vs 0.1) matches the paper.

## Architecture

```
Input volume (256³) → random 96³ crop pair (≥40% overlap)
                          ↓                    ↓
                      view1_aug            view2_aug
                          ↓                    ↓
                  ┌── Online Network ──┐  ┌── Target Network (EMA) ──┐
                  │  Encoder (U-Net)   │  │  Encoder (U-Net)         │
                  │  → GAP → Projector │  │  → GAP → Projector      │
                  │  → Predictor       │  │  (no predictor)          │
                  └────────────────────┘  └──────────────────────────┘
                          ↓                    ↓
                  prediction_1          projection_2 (stop-grad)
                          ↓                    ↓
                      BYOL loss = 2 - 2·cos(pred, proj)
                      (symmetrized: swap views and add)

    After each step: target_params = τ·target + (1-τ)·online
                     τ follows cosine schedule: 0.99 → 1.0
```

**Collapse prevention:** The predictor is only on the online network (asymmetry). The target sees no gradients (stop-gradient + EMA). This asymmetry is what prevents representation collapse — removing the predictor or making the target trainable causes collapse (paper Table 5).

## Checkpoint format

Saved as `.pt` file containing:

```python
{
    'encoder_state_dict': ...,      # for SegmentationUNet3D.from_byol_encoder()
    'online_state_dict': ...,       # full online network
    'target_state_dict': ...,       # full target network
    'optimizer_state_dict': ...,
    'epoch': ...,
    'global_step': ...,
}
```

## Data format

Expects preprocessed `.b2nd` (blosc2) files from the nnssl pipeline. Each file is a z-scored 3D volume (no additional normalization applied — double-normalizing hurts representations). Files may vary in spatial dimensions; the dataloader reads each file's actual shape and computes valid crop positions accordingly.

## TODO

- [ ] Init-time filtering of undersized patches (files smaller than crop_size on any axis)
- [ ] Wire labeled DataLoader into `finetune_segmentation.py`
- [ ] Wandb / tensorboard logging
- [ ] k-NN evaluation during pretraining

## References

- Grill, J.-B., et al. "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning." NeurIPS 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
- Original implementation: [deepmind-research/byol](https://github.com/deepmind/deepmind-research/tree/master/byol)
