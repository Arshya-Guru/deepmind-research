# ============================================================================
# byol3d/utils/dataset_seg_blosc2.py  (REPLACE the previous version)
# ============================================================================

"""
Segmentation dataset with foreground oversampling for sparse Aβ plaques.

Critical design: 50% of crops are centered on foreground voxels, 50% random.
Without this, most crops are all-background and training stalls.

This mirrors nnU-Net's oversampling strategy (they use 33%, we use 50%
because plaques are extremely sparse in these volumes).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

try:
    import blosc2
    HAS_BLOSC2 = True
except ImportError:
    HAS_BLOSC2 = False


# ---------------------------------------------------------------------------
# Augmentations
# ---------------------------------------------------------------------------

class SegAugment3D:
    """Spatial transforms applied identically to image+mask.
    Intensity transforms applied only to image.
    """

    def __init__(
        self,
        p_flip: float = 0.5,
        p_rot90: float = 0.5,
        p_noise: float = 0.15,
        noise_std: float = 0.1,
        p_intensity: float = 0.15,
        scale_range: Tuple[float, float] = (0.9, 1.1),
        shift_range: Tuple[float, float] = (-0.1, 0.1),
        p_elastic: float = 0.2,
    ):
        self.p_flip = p_flip
        self.p_rot90 = p_rot90
        self.p_noise = p_noise
        self.noise_std = noise_std
        self.p_intensity = p_intensity
        self.scale_range = scale_range
        self.shift_range = shift_range
        # elastic deformation helps a lot with only 34 volumes
        self.p_elastic = p_elastic

    def _elastic_deform(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple elastic-like deformation via random affine.

        True elastic deformation is expensive; a small random affine
        (slight rotation + scaling) gives 80% of the benefit.
        """
        # Random small rotation angles (±15°) and scaling (0.85–1.15)
        import math
        angle = (torch.rand(1).item() - 0.5) * 30  # ±15 degrees
        scale = 0.85 + torch.rand(1).item() * 0.3
        rad = math.radians(angle)
        cos_a, sin_a = math.cos(rad), math.sin(rad)

        # 3D affine: rotate in a random plane, scale uniformly
        plane = int(torch.randint(0, 3, (1,)).item())
        theta = torch.eye(3, 4).unsqueeze(0)  # [1, 3, 4]

        if plane == 0:  # rotate in HW plane
            theta[0, 1, 1] = cos_a * scale
            theta[0, 1, 2] = -sin_a * scale
            theta[0, 2, 1] = sin_a * scale
            theta[0, 2, 2] = cos_a * scale
            theta[0, 0, 0] = scale
        elif plane == 1:  # rotate in DW plane
            theta[0, 0, 0] = cos_a * scale
            theta[0, 0, 2] = -sin_a * scale
            theta[0, 2, 0] = sin_a * scale
            theta[0, 2, 2] = cos_a * scale
            theta[0, 1, 1] = scale
        else:  # rotate in DH plane
            theta[0, 0, 0] = cos_a * scale
            theta[0, 0, 1] = -sin_a * scale
            theta[0, 1, 0] = sin_a * scale
            theta[0, 1, 1] = cos_a * scale
            theta[0, 2, 2] = scale

        # Need 5D input for grid_sample: [B, C, D, H, W]
        img_5d = image.unsqueeze(0)  # [1, 1, D, H, W]
        mask_5d = mask.unsqueeze(0)

        grid = torch.nn.functional.affine_grid(
            theta, img_5d.shape, align_corners=False)
        img_out = torch.nn.functional.grid_sample(
            img_5d, grid, mode='bilinear', align_corners=False,
            padding_mode='border')
        mask_out = torch.nn.functional.grid_sample(
            mask_5d, grid, mode='nearest', align_corners=False,
            padding_mode='zeros')  # zeros padding for mask

        return img_out.squeeze(0), mask_out.squeeze(0)

    def __call__(
        self, image: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # --- Spatial (applied identically) ---
        for axis in [1, 2, 3]:
            if torch.rand(1).item() < self.p_flip:
                image = torch.flip(image, [axis])
                mask = torch.flip(mask, [axis])

        if torch.rand(1).item() < self.p_rot90:
            k = int(torch.randint(1, 4, (1,)).item())
            plane = int(torch.randint(0, 3, (1,)).item())
            dims = [(1, 2), (1, 3), (2, 3)][plane]
            image = torch.rot90(image, k, dims)
            mask = torch.rot90(mask, k, dims)

        if torch.rand(1).item() < self.p_elastic:
            image, mask = self._elastic_deform(image, mask)

        # --- Intensity (image only) ---
        if torch.rand(1).item() < self.p_noise:
            image = image + torch.randn_like(image) * self.noise_std

        if torch.rand(1).item() < self.p_intensity:
            s = torch.empty(1).uniform_(*self.scale_range).item()
            o = torch.empty(1).uniform_(*self.shift_range).item()
            image = image * s + o

        return image, mask


# ---------------------------------------------------------------------------
# Dataset with foreground oversampling
# ---------------------------------------------------------------------------

class LightsheetSegDataset(Dataset):
    """Segmentation dataset with foreground oversampling.

    On __init__, pre-scans each mask to cache foreground voxel coordinates.
    At sample time:
      - 50% chance: center crop on a random foreground voxel
      - 50% chance: fully random crop

    This ensures the model always sees plaques during training,
    critical when foreground is <5% of volume.
    """

    def __init__(
        self,
        data_dir: str,
        file_indices: Optional[List[int]] = None,
        crop_size: int = 128,
        augment: bool = True,
        crops_per_volume: int = 8,
        fg_oversample_ratio: float = 0.5,
    ):
        super().__init__()
        self.crop_size = crop_size
        self.augment_fn = SegAugment3D() if augment else None
        self.crops_per_volume = crops_per_volume
        self.fg_ratio = fg_oversample_ratio

        # Find volume directory
        data_path = Path(data_dir)
        spacing_dirs = list(data_path.glob('Spacing__*'))
        vol_dir = spacing_dirs[0] if spacing_dirs else data_path

        # Collect image/mask pairs
        all_images = sorted(vol_dir.glob('patchvolume_[0-9]*.b2nd'))
        all_images = [p for p in all_images if '_seg' not in p.stem]

        if file_indices is not None:
            idx_set = set(file_indices)
            self.image_paths = []
            for p in all_images:
                try:
                    idx = int(p.stem.split('_')[-1])
                    if idx in idx_set:
                        self.image_paths.append(p)
                except ValueError:
                    continue
        else:
            self.image_paths = all_images

        self.mask_paths = []
        for img_p in self.image_paths:
            seg_p = img_p.parent / f"{img_p.stem}_seg{img_p.suffix}"
            if not seg_p.exists():
                raise FileNotFoundError(f"Mask not found: {seg_p}")
            self.mask_paths.append(seg_p)

        # --- Pre-scan masks for foreground coordinates ---
        # This is a one-time cost at dataset creation (~30s for 34 volumes)
        self.fg_coords: List[np.ndarray] = []
        self.spatial_shapes: List[Tuple[int, int, int]] = []
        print(f"Pre-scanning {len(self.mask_paths)} masks for foreground...")

        for i, mask_p in enumerate(self.mask_paths):
            arr = blosc2.open(str(mask_p), mode='r', mmap_mode='r')
            shape = self._get_spatial_shape(arr.shape)
            self.spatial_shapes.append(shape)

            # Read full mask to find foreground
            mask_vol = np.array(arr[:], dtype=np.float32, copy=True)
            while mask_vol.ndim > 3 and mask_vol.shape[0] == 1:
                mask_vol = mask_vol.squeeze(0)
            fg = np.argwhere(mask_vol > 0.5)  # [N_fg, 3]

            self.fg_coords.append(fg)
            fg_frac = len(fg) / np.prod(shape) * 100
            if i < 5 or i == len(self.mask_paths) - 1:
                print(f"  vol {i:03d}: shape={shape}, "
                      f"fg_voxels={len(fg):,} ({fg_frac:.2f}%)")

        total_fg = sum(len(c) for c in self.fg_coords)
        total_vox = sum(np.prod(s) for s in self.spatial_shapes)
        print(f"  Total: {total_fg:,} fg voxels / {total_vox:,} "
              f"({total_fg/total_vox*100:.2f}%)")
        print(f"LightsheetSegDataset: {len(self.image_paths)} volumes, "
              f"crop={crop_size}³, crops_per_vol={crops_per_volume}, "
              f"fg_oversample={fg_oversample_ratio:.0%}")

    def __len__(self) -> int:
        return len(self.image_paths) * self.crops_per_volume

    @staticmethod
    def _get_spatial_shape(raw_shape: tuple) -> Tuple[int, int, int]:
        shape = list(raw_shape)
        while len(shape) > 3 and shape[0] == 1:
            shape = shape[1:]
        return tuple(shape)

    @staticmethod
    def _read_crop(arr, pos, crop_size) -> np.ndarray:
        c = crop_size
        d, h, w = pos
        if arr.ndim == 3:
            return np.array(arr[d:d+c, h:h+c, w:w+c],
                            dtype=np.float32, copy=True)
        elif arr.ndim == 4 and arr.shape[0] == 1:
            return np.array(arr[0, d:d+c, h:h+c, w:w+c],
                            dtype=np.float32, copy=True)
        elif arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] == 1:
            return np.array(arr[0, 0, d:d+c, h:h+c, w:w+c],
                            dtype=np.float32, copy=True)
        raise ValueError(f"Unexpected shape {arr.shape}")

    def _compute_crop_position(
        self, vol_idx: int
    ) -> Tuple[int, int, int]:
        """Pick a crop position, biased toward foreground."""
        spatial = self.spatial_shapes[vol_idx]
        c = self.crop_size
        max_d, max_h, max_w = [s - c for s in spatial]

        fg = self.fg_coords[vol_idx]
        use_fg = (len(fg) > 0 and np.random.rand() < self.fg_ratio)

        if use_fg:
            # Pick a random foreground voxel, center crop on it
            idx = np.random.randint(len(fg))
            center = fg[idx]  # [d, h, w]
            d = int(np.clip(center[0] - c // 2, 0, max_d))
            h = int(np.clip(center[1] - c // 2, 0, max_h))
            w = int(np.clip(center[2] - c // 2, 0, max_w))
        else:
            d = int(np.random.randint(0, max_d + 1))
            h = int(np.random.randint(0, max_h + 1))
            w = int(np.random.randint(0, max_w + 1))

        return (d, h, w)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        vol_idx = idx % len(self.image_paths)
        try:
            return self._load_sample(vol_idx)
        except Exception:
            fallback = int(np.random.randint(0, len(self.image_paths)))
            try:
                return self._load_sample(fallback)
            except Exception:
                c = self.crop_size
                return {'image': torch.zeros(1, c, c, c),
                        'mask': torch.zeros(1, c, c, c)}

    def _load_sample(self, vol_idx: int) -> Dict[str, torch.Tensor]:
        pos = self._compute_crop_position(vol_idx)
        c = self.crop_size

        img_arr = blosc2.open(str(self.image_paths[vol_idx]),
                              mode='r', mmap_mode='r')
        mask_arr = blosc2.open(str(self.mask_paths[vol_idx]),
                               mode='r', mmap_mode='r')

        img_crop = self._read_crop(img_arr, pos, c)
        mask_crop = self._read_crop(mask_arr, pos, c)

        img_t = torch.from_numpy(
            np.ascontiguousarray(img_crop)).unsqueeze(0).clone().float()
        mask_t = torch.from_numpy(
            np.ascontiguousarray(mask_crop)).unsqueeze(0).clone().float()
        mask_t = (mask_t > 0.5).float()

        if self.augment_fn is not None:
            img_t, mask_t = self.augment_fn(img_t, mask_t)

        return {'image': img_t, 'mask': mask_t}


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------

def load_nnunet_splits(
    dataset_dir: str, fold: int = 0
) -> Tuple[List[int], List[int]]:
    splits_path = Path(dataset_dir) / 'splits_final.json'
    if not splits_path.exists():
        raise FileNotFoundError(f"No splits_final.json at {splits_path}")
    with open(splits_path) as f:
        splits = json.load(f)
    if fold >= len(splits):
        raise ValueError(
            f"Fold {fold} not in splits (have {len(splits)} folds)")
    split = splits[fold]

    def extract_indices(keys):
        indices = []
        for k in keys:
            try:
                indices.append(int(k.split('_')[-1]))
            except ValueError:
                continue
        return sorted(indices)

    train_idx = extract_indices(split['train'])
    val_idx = extract_indices(split['val'])
    print(f"Split fold {fold}: {len(train_idx)} train, {len(val_idx)} val")
    return train_idx, val_idx


def _seg_worker_init_fn(worker_id: int):
    if HAS_BLOSC2:
        blosc2.set_nthreads(1)
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


def create_seg_dataloaders(
    data_dir: str,
    fold: int = 0,
    crop_size: int = 128,
    batch_size: int = 2,
    num_workers: int = 4,
    crops_per_volume: int = 8,
    pin_memory: bool = True,
) -> Tuple[DataLoader, DataLoader]:
    train_idx, val_idx = load_nnunet_splits(data_dir, fold=fold)

    train_ds = LightsheetSegDataset(
        data_dir=data_dir, file_indices=train_idx,
        crop_size=crop_size, augment=True,
        crops_per_volume=crops_per_volume)

    val_ds = LightsheetSegDataset(
        data_dir=data_dir, file_indices=val_idx,
        crop_size=crop_size, augment=False,
        crops_per_volume=2, fg_oversample_ratio=0.0)  # val: random only

    kw = dict(num_workers=num_workers, pin_memory=pin_memory,
              persistent_workers=False, worker_init_fn=_seg_worker_init_fn)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        drop_last=True, **kw)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        drop_last=False, **kw)
    return train_loader, val_loader