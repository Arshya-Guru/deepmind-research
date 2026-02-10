# byol3d

BYOL (Bootstrap Your Own Latent) self-supervised pretraining adapted for 3D lightsheet microscopy, targeting Aβ plaque segmentation.

Reimplemented in PyTorch from [DeepMind's original JAX/Haiku codebase](https://github.com/deepmind/deepmind-research/tree/master/byol) (Grill et al., 2020). The original 2D ResNet + ImageNet pipeline is replaced with a 3D U-Net encoder operating on single-channel fluorescence volumes.

## Why BYOL over SimCLR

- **No negative pairs** → robust at small batch sizes (2–16). SimCLR performance collapses below batch 256.
- **Augmentation tolerant** → BYOL degrades ~3 pts with fewer augmentations vs ~10 pts for SimCLR (paper Table 5).
- **3D memory constraints** → 96³ crops × batch 16 already pushes 20–30 GB VRAM. BYOL doesn't need the massive batches SimCLR requires.

## Architecture

```
256³ patch → [random 96³ crop pair, overlap ≥ 0.4] → augment independently
                                                          ↓
                                              ┌──── view 1 ────┐   ┌──── view 2 ────┐
                                              │                │   │                │
                                          Online net       Target net (EMA)
                                              │                │   │                │
                                          Encoder f_θ      Encoder f_ξ
                                          (3D U-Net         (3D U-Net
                                          contracting)      contracting)
                                              │                │
                                          GAP → 320-d      GAP → 320-d
                                              │                │
                                          Projector g_θ    Projector g_ξ
                                          (MLP 320→2048→256) (MLP, same arch)
                                              │                │
                                          Predictor q_θ       ✗  (asymmetric)
                                          (MLP 256→2048→256)
                                              │                │
                                              └──── L2 loss ───┘
                                                (symmetrized)
```

**Encoder:** 5-level 3D U-Net contracting path `[32, 64, 128, 256, 320]` channels with residual blocks, InstanceNorm3d, LeakyReLU. After global average pooling the bottleneck gives a 320-dim representation.

**Projector/Predictor:** 2-layer MLPs with BatchNorm1d + ReLU (matching paper's depth-2 default from Table 14). The predictor only exists on the online branch — this asymmetry is what prevents collapse.

**Target network:** exponential moving average of the online network, with a cosine schedule that ramps τ from 0.99 → 1.0 over training.

**Param counts (real config):**

| Component | Params |
|-----------|--------|
| Online network (encoder + projector + predictor) | ~10.9M |
| Target network (encoder + projector) | ~9.8M |
| Segmentation U-Net (encoder + decoder) | ~16.8M |

## Setup

Requires [pixi](https://pixi.sh). The `pixi.toml` lives in the repo root (`deepmind-research/`).

```bash
cd deepmind-research/

# GPU environment (cluster with CUDA 12+)
pixi install

# CPU-only environment (local dev, macOS)
pixi install -e cpu

# Dev environment (GPU + ipython + tensorboard)
pixi install -e dev
```

Or without pixi — just install the deps manually:

```bash
pip install torch blosc2 numpy scipy
```

## Quick start

```bash
# 1. Verify everything works (synthetic data, no GPU, ~5 seconds)
pixi run smoke

# 2. Run unit tests on network shapes / gradient flow
pixi run test

# 3. Pretrain on your data
pixi run pretrain

# 4. Fine-tune from checkpoint
pixi run finetune
```

## CLI reference

### Pretraining

```
python -m byol3d.byol_pretrain_3d [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | `smoke` | Config preset: `full` (cluster) or `smoke` (local test) |
| `--epochs` | 300 (full) / 2 (smoke) | Number of training epochs |
| `--batch-size` | 4 (full) / 2 (smoke) | Per-GPU batch size |
| `--data-dir` | see config | Path to directory of `.b2nd` or `.npy` patch files |
| `--save-dir` | `/tmp/byol3d_checkpoints` | Where to save checkpoints |

**Examples:**

```bash
# Smoke test — synthetic data, tiny model, 4 steps
python -m byol3d.byol_pretrain_3d --config smoke

# Full training with defaults (300 epochs, batch 4)
python -m byol3d.byol_pretrain_3d --config full \
    --data-dir /nfs/khan/trainees/apooladi/abeta/nnssl_data/8/nnssl_data/preprocessed

# Override everything
python -m byol3d.byol_pretrain_3d --config full \
    --epochs 500 \
    --batch-size 16 \
    --data-dir /path/to/my/patches \
    --save-dir ./my_checkpoints

# Short run to verify GPU training works
python -m byol3d.byol_pretrain_3d --config full \
    --epochs 5 \
    --batch-size 2 \
    --data-dir /path/to/patches \
    --save-dir /tmp/test_run
```

The `--config full` preset loads defaults from `byol3d/configs/lightsheet_3d.py`. Any flag you pass overrides the corresponding config value. If you don't pass `--data-dir`, it uses the hardcoded NFS path in the config.

### Fine-tuning

```
python -m byol3d.finetune_segmentation [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | None | Path to BYOL pretraining `.pt` file. If omitted, trains from scratch. |
| `--data-dir` | None | Path to labeled data (not yet wired — see below) |
| `--freeze-encoder` | False | If set, freezes encoder weights (linear probe / decoder-only training) |
| `--epochs` | 100 | Number of fine-tuning epochs |
| `--batch-size` | 2 | Per-GPU batch size |
| `--lr` | 1e-3 | Base learning rate |

**Examples:**

```bash
# Fine-tune with pretrained encoder (full fine-tuning)
python -m byol3d.finetune_segmentation \
    --checkpoint ./checkpoints/byol3d_pretrain.pt \
    --epochs 200 \
    --lr 5e-4

# Linear probe: freeze encoder, only train decoder
python -m byol3d.finetune_segmentation \
    --checkpoint ./checkpoints/byol3d_pretrain.pt \
    --freeze-encoder \
    --epochs 100

# Train from scratch (no pretraining, baseline comparison)
python -m byol3d.finetune_segmentation \
    --epochs 200 \
    --batch-size 2
```

When `--freeze-encoder` is off (default), the encoder trains with 0.1× the base learning rate (differential LR) while the decoder trains at full LR. This is a standard fine-tuning strategy that preserves pretrained features.

### Pixi tasks

All tasks defined in `pixi.toml`:

| Task | What it does |
|------|--------------|
| `pixi run smoke` | Smoke test: synthetic data, no GPU, ~5 sec |
| `pixi run test` | Network unit tests (shapes, gradients, EMA, loss) |
| `pixi run pretrain` | Full pretraining: batch=16, 300 epochs |
| `pixi run pretrain-short` | Shorter pretraining: batch=16, 100 epochs |
| `pixi run finetune` | Seg fine-tuning from `./checkpoints/byol3d_pretrain.pt` |
| `pixi run finetune-frozen` | Same but with frozen encoder |

To bypass pixi tasks and pass your own flags directly:

```bash
pixi run -- python -m byol3d.byol_pretrain_3d --config full --batch-size 32 --epochs 1000
```

## Config

All hyperparameters live in `byol3d/configs/lightsheet_3d.py`. The `get_config()` function returns a dict. CLI flags override values in this dict at runtime.

If you want to change something that doesn't have a CLI flag (e.g. encoder depth, projector dims, augmentation params), edit the config file directly:

```python
# byol3d/configs/lightsheet_3d.py

# Change encoder to 4 levels instead of 5
encoder=dict(
    in_channels=1,
    base_channels=32,
    num_levels=4,        # was 5
    channels=None,
    use_residuals=True,
),

# Smaller projector
projector=dict(
    hidden_dim=1024,     # was 2048
    output_dim=128,      # was 256
),
```

Augmentation parameters are in `byol3d/utils/augmentations_3d.py` in the `augment_config` dict at the top of the file. The asymmetry between view1 and view2 (blur probability 0.5 vs 0.1) matches the paper.

## Data format

The dataset loader expects a flat directory of 3D volumes as either:

- **`.b2nd`** files (blosc2 compressed, from nnssl pipeline) — preferred, supports partial decompression
- **`.npy`** files (numpy arrays) — fallback

Each file should be a single 3D volume (e.g. 256×256×256). The loader handles normalization to [0, 1], crop extraction, and augmentation automatically.

The data directory is set via `--data-dir` or hardcoded in the config. Currently pointing to:

```
/nfs/khan/trainees/apooladi/abeta/nnssl_data/8/nnssl_data/preprocessed
```

## VRAM estimates

| Batch size | Crop size | Approx. VRAM | GPU |
|------------|-----------|-------------|-----|
| 2 | 96³ | ~8 GB | V100 (16 GB) ✓ |
| 4 | 96³ | ~12 GB | V100 (32 GB) ✓ |
| 8 | 96³ | ~18 GB | A100 (40 GB) ✓ |
| 16 | 96³ | ~28 GB | A100 (80 GB) ✓ |
| 32 | 96³ | ~50 GB | A100 (80 GB), tight |

If you OOM at batch 16, try gradient accumulation (not yet implemented — accumulate gradients over N forward passes before stepping) or drop to 80³ crops.

## Checkpoints

Pretraining saves checkpoints containing:

```python
{
    'epoch': int,
    'global_step': int,
    'online_state_dict': ...,     # full online network (encoder + projector + predictor)
    'target_state_dict': ...,     # full target network
    'encoder_state_dict': ...,    # just the encoder (for fine-tuning)
    'optimizer_state_dict': ...,  # (intermediate checkpoints only)
    'config': dict,               # config used for this run
}
```

The fine-tuning script loads `encoder_state_dict` from the checkpoint and attaches a fresh decoder.

## Project structure

```
deepmind-research/
├── pixi.toml                         # pixi workspace config
├── .gitignore
├── byol/                             # original DeepMind JAX repo (untouched)
│
└── byol3d/                           # 3D PyTorch reimplementation
    ├── __init__.py
    ├── byol_pretrain_3d.py           # pretraining entry point + training loop
    ├── finetune_segmentation.py      # segmentation fine-tuning entry point
    ├── test_networks.py              # unit tests for all network components
    ├── configs/
    │   ├── __init__.py
    │   └── lightsheet_3d.py          # all hyperparameters + presets
    └── utils/
        ├── __init__.py
        ├── networks_3d.py            # encoder, projector, predictor, BYOL wrapper, seg decoder
        ├── augmentations_3d.py       # 3D augmentation pipeline + crop extraction
        └── dataset_blosc2.py         # blosc2/npy data loading + DataLoader factory
```

## TODO

- [ ] **Wire labeled DataLoader into `finetune_segmentation.py`** — the training loop has a placeholder with random tensors. Replace the `# ---- REPLACE THIS ----` section with your real `(image, mask)` DataLoader.
- [ ] **Multi-GPU / DDP** — currently single-GPU. For batch 16+ across multiple GPUs, wrap with `torch.nn.parallel.DistributedDataParallel`.
- [ ] **Gradient accumulation** — for effective batch sizes > physical batch size.
- [ ] **Mixed precision (AMP)** — `torch.cuda.amp` would roughly halve VRAM usage.
- [ ] **Wandb / tensorboard logging** — currently prints to stdout only.
- [ ] **k-NN evaluation during pretraining** — track representation quality without fine-tuning.

## References

- Grill, J.-B., et al. "Bootstrap Your Own Latent: A New Approach to Self-Supervised Learning." NeurIPS 2020. [arXiv:2006.07733](https://arxiv.org/abs/2006.07733)
- Original implementation: [deepmind-research/byol](https://github.com/deepmind/deepmind-research/tree/master/byol)
