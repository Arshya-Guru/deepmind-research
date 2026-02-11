"""
Dataset loader for blosc2-compressed lightsheet patches.

Loads preprocessed patches from the nnssl pipeline, extracts
overlap-constrained crop pairs, and applies BYOL augmentations.

Key design decisions informed by nnssl internals:
  - Data is ALREADY z-scored during nnssl preprocessing — NO normalization here.
  - Blosc2 partial reads: only chunks overlapping each crop are decompressed.
  - blosc2.set_nthreads(1) per DataLoader worker to avoid thread contention.
  - mmap_mode='r' on ALL blosc2.open calls so the OS page cache works.
  - np.array(copy=True) + .clone() on ALL reads to ensure tensors own their
    memory (prevents "Trying to resize storage that is not resizable").
  - persistent_workers=False to avoid stale mmap handles across forks.
  - Single blosc2.open() per sample: shape + crops from one handle.
  - NO global shape cache — files may have different spatial dimensions.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from byol3d.utils.augmentations_3d import (
    Augment3D,
    augment_config,
)


# ---------------------------------------------------------------------------
# Attempt to import blosc2; graceful fallback if unavailable
# ---------------------------------------------------------------------------
try:
    import blosc2
    HAS_BLOSC2 = True
except ImportError:
    HAS_BLOSC2 = False


# ---------------------------------------------------------------------------
# Crop position computation (data-free — only needs volume shape)
# ---------------------------------------------------------------------------

def _constrained_start(ref_start: int, max_offset: int, max_start: int) -> int:
    """Random start position within max_offset of ref_start, clamped to [0, max_start]."""
    lo = max(0, ref_start - max_offset)
    hi = min(max_start, ref_start + max_offset)
    return int(np.random.randint(lo, hi + 1))


def compute_crop_positions(
    spatial_shape: Tuple[int, int, int],
    crop_size: int = 96,
    min_overlap_fraction: float = 0.4,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    """Compute two overlapping crop start positions from volume shape.

    Args:
        spatial_shape: (D, H, W) of the source volume
        crop_size: spatial size of each crop
        min_overlap_fraction: minimum volumetric overlap between crops

    Returns:
        (pos1, pos2): each is (d, h, w) start coordinate
    """
    D, H, W = spatial_shape

    max_d = D - crop_size
    max_h = H - crop_size
    max_w = W - crop_size

    if max_d < 0 or max_h < 0 or max_w < 0:
        raise ValueError(
            f"Volume spatial dims {spatial_shape} too small for crop_size={crop_size}")

    # First crop: random position
    d1 = int(np.random.randint(0, max_d + 1))
    h1 = int(np.random.randint(0, max_h + 1))
    w1 = int(np.random.randint(0, max_w + 1))

    # Second crop: constrained to overlap with first
    per_axis_min = min_overlap_fraction ** (1.0 / 3.0)
    max_offset = int(crop_size * (1.0 - per_axis_min))

    d2 = _constrained_start(d1, max_offset, max_d)
    h2 = _constrained_start(h1, max_offset, max_h)
    w2 = _constrained_start(w1, max_offset, max_w)

    return (d1, h1, w1), (d2, h2, w2)


# ---------------------------------------------------------------------------
# Worker init function (critical for blosc2 + DataLoader multiprocessing)
# ---------------------------------------------------------------------------

def _worker_init_fn(worker_id: int) -> None:
    """Per-worker initialization for DataLoader multiprocessing."""
    if HAS_BLOSC2:
        blosc2.set_nthreads(1)
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LightsheetBYOLDataset(Dataset):
    """Dataset for BYOL pretraining on blosc2-compressed lightsheet patches.

    Each sample returns two augmented views (crop pairs) from one patch.

    IMPORTANT: The .b2nd data is already z-scored by nnssl preprocessing.
    No additional normalization is applied.
    """

    def __init__(
        self,
        data_dir: str,
        crop_size: int = 96,
        min_overlap: float = 0.4,
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.crop_size = crop_size
        self.min_overlap = min_overlap
        self.cache_dir = Path(cache_dir) if cache_dir else None

        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Scan for data files
        data_path = Path(data_dir)
        self.file_paths: List[Path] = sorted(
            list(data_path.rglob('*.b2nd')) + list(data_path.rglob('*.npy'))
        )
        if not self.file_paths:
            raise FileNotFoundError(f"No .b2nd or .npy files found in {data_dir}")

        print(f"LightsheetBYOLDataset: found {len(self.file_paths)} files")

        # Build augmentations
        self.view1_aug = Augment3D(**augment_config['view1'])
        self.view2_aug = Augment3D(**augment_config['view2'])

        # Print first file info (diagnostic only — NOT used as global cache)
        path = self.file_paths[0]
        if path.suffix == '.b2nd' and HAS_BLOSC2:
            arr = blosc2.open(str(path), mode='r', mmap_mode='r')
            print(f"  First file: {path.name}, shape={arr.shape}, "
                  f"dtype={arr.dtype}, chunks={arr.chunks}")

    def __len__(self) -> int:
        return len(self.file_paths)

    @staticmethod
    def _get_spatial_shape(raw_shape: tuple) -> Tuple[int, int, int]:
        """Extract 3D spatial dims from raw shape, handling leading singletons."""
        shape = list(raw_shape)
        while len(shape) > 3 and shape[0] == 1:
            shape = shape[1:]
        if len(shape) != 3:
            raise ValueError(f"Cannot extract 3D spatial dims from shape {raw_shape}")
        return tuple(shape)

    def _cache_path(self, path: Path) -> Optional[Path]:
        """Get the local cache path for a file, or None if caching disabled."""
        if self.cache_dir is None:
            return None
        return self.cache_dir / (path.stem + '.npy')

    @staticmethod
    def _slice_crops_from_handle(
        arr, spatial_shape: Tuple[int, int, int],
        pos1: Tuple[int, ...], pos2: Tuple[int, ...],
        crop_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Slice two crops from an already-opened blosc2 handle.

        Uses np.array(copy=True) so the returned arrays own their memory
        (not backed by mmap storage — required for torch collation).
        """
        c = crop_size
        d1, h1, w1 = pos1
        d2, h2, w2 = pos2

        if arr.ndim == 3:
            crop1 = np.array(arr[d1:d1+c, h1:h1+c, w1:w1+c], dtype=np.float32, copy=True)
            crop2 = np.array(arr[d2:d2+c, h2:h2+c, w2:w2+c], dtype=np.float32, copy=True)
        elif arr.ndim == 4 and arr.shape[0] == 1:
            crop1 = np.array(arr[0, d1:d1+c, h1:h1+c, w1:w1+c], dtype=np.float32, copy=True)
            crop2 = np.array(arr[0, d2:d2+c, h2:h2+c, w2:w2+c], dtype=np.float32, copy=True)
        elif arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] == 1:
            crop1 = np.array(arr[0, 0, d1:d1+c, h1:h1+c, w1:w1+c], dtype=np.float32, copy=True)
            crop2 = np.array(arr[0, 0, d2:d2+c, h2:h2+c, w2:w2+c], dtype=np.float32, copy=True)
        else:
            raise ValueError(f"Unexpected blosc2 shape {arr.shape}")

        # Paranoid shape check — if this fires, the file is smaller than
        # crop_size on some axis, which compute_crop_positions should have
        # caught. But belt-and-suspenders never hurts.
        assert crop1.shape == (c, c, c), \
            f"crop1 shape {crop1.shape} != expected ({c},{c},{c}), " \
            f"file shape={arr.shape}, spatial={spatial_shape}, pos1={pos1}"
        assert crop2.shape == (c, c, c), \
            f"crop2 shape {crop2.shape} != expected ({c},{c},{c}), " \
            f"file shape={arr.shape}, spatial={spatial_shape}, pos2={pos2}"

        return crop1, crop2

    def _read_crops_cached(
        self, cache_path: Path, pos1: Tuple[int, ...], pos2: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read two crop regions from a cached .npy file via memory-map."""
        vol = np.load(str(cache_path), mmap_mode='r')
        c = self.crop_size
        d1, h1, w1 = pos1
        d2, h2, w2 = pos2
        crop1 = np.array(vol[d1:d1+c, h1:h1+c, w1:w1+c], dtype=np.float32, copy=True)
        crop2 = np.array(vol[d2:d2+c, h2:h2+c, w2:w2+c], dtype=np.float32, copy=True)
        return crop1, crop2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a volume and return two augmented crop views."""
        try:
            return self._load_sample(idx)
        except Exception as e:
            # Fallback: pick a random different sample
            fallback_idx = int(np.random.randint(0, len(self)))
            try:
                return self._load_sample(fallback_idx)
            except Exception:
                # Last resort: return zeros with .clone() for owned storage
                c = self.crop_size
                zeros = torch.zeros(1, c, c, c)
                return {'view1': zeros.clone(), 'view2': zeros.clone(), 'index': idx}

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.file_paths[idx]
        cache_path = self._cache_path(path)

        # =================================================================
        # CORE FIX: One blosc2.open() per sample. Read shape AND crops
        # from the SAME handle. No global shape cache — files may differ.
        # =================================================================

        if cache_path is not None and cache_path.exists():
            # --- Cached path: read npy header for REAL shape, then slice ---
            with open(cache_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                raw_shape, _, _ = np.lib.format._read_array_header(f, version)
            spatial_shape = self._get_spatial_shape(raw_shape)
            pos1, pos2 = compute_crop_positions(
                spatial_shape, self.crop_size, self.min_overlap)
            crop1, crop2 = self._read_crops_cached(cache_path, pos1, pos2)

        elif path.suffix == '.b2nd' and HAS_BLOSC2:
            # --- Single open: shape + crops from one handle ---
            arr = blosc2.open(str(path), mode='r', mmap_mode='r')
            spatial_shape = self._get_spatial_shape(arr.shape)
            pos1, pos2 = compute_crop_positions(
                spatial_shape, self.crop_size, self.min_overlap)

            if cache_path is not None and not cache_path.exists():
                # First epoch with caching: read full, extract, save
                vol = np.array(arr[:], dtype=np.float32, copy=True)
                while vol.ndim > 3 and vol.shape[0] == 1:
                    vol = vol.squeeze(0)
                c = self.crop_size
                d1, h1, w1 = pos1
                d2, h2, w2 = pos2
                crop1 = vol[d1:d1+c, h1:h1+c, w1:w1+c].copy()
                crop2 = vol[d2:d2+c, h2:h2+c, w2:w2+c].copy()
                try:
                    np.save(str(cache_path), vol)
                except (OSError, Exception):
                    pass
            else:
                # No caching: partial reads (~11% of volume)
                crop1, crop2 = self._slice_crops_from_handle(
                    arr, spatial_shape, pos1, pos2, self.crop_size)

        elif path.suffix == '.npy':
            # --- .npy fallback ---
            vol = np.load(str(path)).astype(np.float32)
            while vol.ndim > 3 and vol.shape[0] == 1:
                vol = vol.squeeze(0)
            spatial_shape = vol.shape[:3]
            pos1, pos2 = compute_crop_positions(
                spatial_shape, self.crop_size, self.min_overlap)
            c = self.crop_size
            d1, h1, w1 = pos1
            d2, h2, w2 = pos2
            crop1 = vol[d1:d1+c, h1:h1+c, w1:w1+c].copy()
            crop2 = vol[d2:d2+c, h2:h2+c, w2:w2+c].copy()
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # --- To tensor: ensure owned, contiguous memory ---
        crop1 = np.ascontiguousarray(crop1, dtype=np.float32)
        crop2 = np.ascontiguousarray(crop2, dtype=np.float32)
        crop1_t = torch.from_numpy(crop1).unsqueeze(0).clone()  # [1, D, H, W]
        crop2_t = torch.from_numpy(crop2).unsqueeze(0).clone()

        # Final shape assertion — catch ANY mismatch before collation
        c = self.crop_size
        expected = (1, c, c, c)
        assert crop1_t.shape == expected, \
            f"view1 shape {crop1_t.shape} != {expected}, file={path.name}"
        assert crop2_t.shape == expected, \
            f"view2 shape {crop2_t.shape} != {expected}, file={path.name}"

        # --- Apply augmentations ---
        view1 = self.view1_aug(crop1_t.unsqueeze(0)).squeeze(0)
        view2 = self.view2_aug(crop2_t.unsqueeze(0)).squeeze(0)

        return {
            'view1': view1,
            'view2': view2,
            'index': idx,
        }


# ---------------------------------------------------------------------------
# Synthetic dataset for testing without real data
# ---------------------------------------------------------------------------

class SyntheticBYOLDataset(Dataset):
    """Synthetic dataset for testing the pipeline without real data."""

    def __init__(
        self,
        num_samples: int = 100,
        volume_size: int = 32,
        crop_size: int = 16,
        min_overlap: float = 0.4,
        view1_aug: Optional[Augment3D] = None,
        view2_aug: Optional[Augment3D] = None,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.volume_size = volume_size
        self.crop_size = crop_size
        self.min_overlap = min_overlap
        self.view1_aug = view1_aug or Augment3D(**augment_config['view1'])
        self.view2_aug = view2_aug or Augment3D(**augment_config['view2'])

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gen = np.random.RandomState(idx)
        vol = gen.rand(self.volume_size, self.volume_size, self.volume_size).astype(np.float32)

        pos1, pos2 = compute_crop_positions(
            (self.volume_size, self.volume_size, self.volume_size),
            self.crop_size, self.min_overlap)

        c = self.crop_size
        d1, h1, w1 = pos1
        d2, h2, w2 = pos2
        crop1 = torch.from_numpy(vol[d1:d1+c, h1:h1+c, w1:w1+c].copy()).float().unsqueeze(0)
        crop2 = torch.from_numpy(vol[d2:d2+c, h2:h2+c, w2:w2+c].copy()).float().unsqueeze(0)

        view1 = self.view1_aug(crop1.unsqueeze(0)).squeeze(0)
        view2 = self.view2_aug(crop2.unsqueeze(0)).squeeze(0)

        return {'view1': view1, 'view2': view2, 'index': idx}


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_byol_dataloader(
    data_dir: Optional[str] = None,
    batch_size: int = 4,
    crop_size: int = 96,
    min_overlap: float = 0.4,
    num_workers: int = 4,
    pin_memory: bool = True,
    synthetic: bool = False,
    synthetic_num_samples: int = 100,
    synthetic_volume_size: int = 32,
    cache_dir: Optional[str] = None,
) -> DataLoader:
    """Create a DataLoader for BYOL pretraining."""
    if synthetic or data_dir is None:
        crop = min(crop_size, synthetic_volume_size // 2)
        dataset = SyntheticBYOLDataset(
            num_samples=synthetic_num_samples,
            volume_size=synthetic_volume_size,
            crop_size=crop,
        )
    else:
        dataset = LightsheetBYOLDataset(
            data_dir=data_dir,
            crop_size=crop_size,
            min_overlap=min_overlap,
            cache_dir=cache_dir,
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=False,
        prefetch_factor=2 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )