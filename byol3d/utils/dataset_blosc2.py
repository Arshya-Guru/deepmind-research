"""
Dataset loader for blosc2-compressed lightsheet patches.

Loads preprocessed 256^3 patches from the nnssl pipeline, extracts
overlap-constrained crop pairs, and applies BYOL augmentations.

Key design decisions informed by nnssl internals:
  - Data is ALREADY z-scored during nnssl preprocessing — NO normalization here.
    The .b2nd values are the final training values. Double-normalizing (e.g.
    rescaling to [0,1]) compresses the learned distribution and hurts
    representation quality.
  - Blosc2 partial reads: we open a handle (no decompression), compute crop
    positions from shape metadata, then slice the handle — only the chunks
    overlapping each 96^3 crop are decompressed. For two 96^3 crops from a
    256^3 volume this decompresses ~11% of the data instead of 100%.
  - blosc2.set_nthreads(1) per DataLoader worker to avoid thread contention
    across multiprocessing workers.
  - Local SSD cache: first epoch reads from NFS + blosc2, saves decompressed
    .npy to local disk. Subsequent epochs use numpy mmap for fast random access.
  - Error handling: corrupted files silently fall back to a random other sample
    (matches nnssl behavior).
  - The .pkl sidecar contains spatial metadata only (spacing, bbox, shape) —
    not needed for training, so we don't load it.

Data flow:
  1. Scan directory recursively for .b2nd files
  2. On __getitem__:
     a. Open blosc2 handle (lazy, no decompression)
     b. Compute two overlapping 96^3 crop positions from volume shape
     c. Partial-read each crop region (only relevant chunks decompressed)
     d. Apply view1/view2 augmentations independently
  3. Return dict with 'view1' and 'view2' tensors
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

    This is the data-free counterpart of extract_overlapping_crops() from
    augmentations_3d.py. It returns positions only, so we can slice a blosc2
    handle or numpy mmap without loading the full volume.

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
    """Per-worker initialization for DataLoader multiprocessing.

    - Sets blosc2 to single-threaded (avoids thread contention across workers)
    - Seeds numpy RNG per worker (avoids identical augmentations)
    """
    if HAS_BLOSC2:
        blosc2.set_nthreads(1)
    # Unique seed per worker per epoch (PyTorch sets base seed per epoch)
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed + worker_id)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class LightsheetBYOLDataset(Dataset):
    """Dataset for BYOL pretraining on blosc2-compressed lightsheet patches.

    Each sample returns two augmented views (crop pairs) from one 256^3 patch.

    IMPORTANT: The .b2nd data is already z-scored by nnssl preprocessing.
    No additional normalization is applied. The voxel values in the .b2nd
    files are the final training values.

    Args:
        data_dir: path to directory containing .b2nd files
        crop_size: spatial size of each crop (96 for our 3D U-Net)
        min_overlap: minimum volumetric overlap fraction between crop pairs
        view1_aug: augmentation pipeline for view 1
        view2_aug: augmentation pipeline for view 2
        file_ext: file extension to search for
        cache_dir: local SSD directory for caching decompressed volumes.
            First epoch: read from NFS blosc2, save .npy to cache.
            Subsequent epochs: numpy mmap from local SSD (~10x faster).
    """

    def __init__(
        self,
        data_dir: str,
        crop_size: int = 96,
        min_overlap: float = 0.4,
        view1_aug: Optional[Augment3D] = None,
        view2_aug: Optional[Augment3D] = None,
        file_ext: str = '.b2nd',
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.min_overlap = min_overlap
        self.file_ext = file_ext

        # Local SSD cache for decompressed volumes
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Default augmentations from our config
        self.view1_aug = view1_aug or Augment3D(**augment_config['view1'])
        self.view2_aug = view2_aug or Augment3D(**augment_config['view2'])

        # Discover files — recursive glob for nnssl nested structure:
        #   preprocessed/Dataset001_Pretrain/nnsslPlans_noresample/
        #       PatchCollection/1/patch_00000/ses-01/patch_00000.b2nd
        self.file_paths = sorted(self.data_dir.rglob(f'*{file_ext}'))
        if not self.file_paths:
            self.file_paths = sorted(self.data_dir.rglob('*.npy'))
            if self.file_paths:
                self.file_ext = '.npy'

        if not self.file_paths:
            raise FileNotFoundError(
                f"No {file_ext} or .npy files found in {data_dir} "
                f"(searched recursively)")

        print(f"LightsheetBYOLDataset: found {len(self.file_paths)} files")

        # Probe first file for shape info (blosc2 metadata, no decompression)
        self._log_first_file()

        # Set blosc2 threads for main process too
        if HAS_BLOSC2:
            blosc2.set_nthreads(1)

    def _log_first_file(self):
        """Log metadata about the first file (shape, dtype) without full decompression."""
        path = self.file_paths[0]
        if path.suffix == '.b2nd' and HAS_BLOSC2:
            arr = blosc2.open(str(path), mode='r')
            print(f"  First file: {path.name}, shape={arr.shape}, "
                  f"dtype={arr.dtype}, chunks={arr.chunks}")
        elif path.suffix == '.npy':
            # Read just the header
            with open(path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                shape, _, dtype = np.lib.format._read_array_header(f, version)
            print(f"  First file: {path.name}, shape={shape}, dtype={dtype}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def _get_spatial_shape(self, raw_shape: tuple) -> Tuple[int, int, int]:
        """Extract 3D spatial dims from raw shape, handling leading singletons.

        Handles: (D,H,W), (1,D,H,W), (1,1,D,H,W)
        """
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

    def _read_crops_blosc2(
        self, path: Path, pos1: Tuple[int, ...], pos2: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read two crop regions from a blosc2 file via partial decompression.

        Only the chunks overlapping each crop are decompressed. For 96^3 crops
        from 256^3 volumes, this decompresses ~11% of the volume instead of 100%.
        """
        arr = blosc2.open(str(path), mode='r')
        c = self.crop_size
        d1, h1, w1 = pos1
        d2, h2, w2 = pos2

        # Determine indexing based on array shape (handle leading singleton dims)
        if arr.ndim == 3:
            crop1 = np.asarray(arr[d1:d1+c, h1:h1+c, w1:w1+c], dtype=np.float32)
            crop2 = np.asarray(arr[d2:d2+c, h2:h2+c, w2:w2+c], dtype=np.float32)
        elif arr.ndim == 4 and arr.shape[0] == 1:
            crop1 = np.asarray(arr[0, d1:d1+c, h1:h1+c, w1:w1+c], dtype=np.float32)
            crop2 = np.asarray(arr[0, d2:d2+c, h2:h2+c, w2:w2+c], dtype=np.float32)
        elif arr.ndim == 5 and arr.shape[0] == 1 and arr.shape[1] == 1:
            crop1 = np.asarray(arr[0, 0, d1:d1+c, h1:h1+c, w1:w1+c], dtype=np.float32)
            crop2 = np.asarray(arr[0, 0, d2:d2+c, h2:h2+c, w2:w2+c], dtype=np.float32)
        else:
            raise ValueError(f"Unexpected blosc2 shape {arr.shape} in {path}")

        return crop1, crop2

    def _read_crops_cached(
        self, cache_path: Path, pos1: Tuple[int, ...], pos2: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read two crop regions from a cached .npy file via memory-map.

        The mmap means only the pages covering the crop are read from disk —
        fast random access from local SSD without loading the full 64MB volume.
        """
        vol = np.load(str(cache_path), mmap_mode='r')
        c = self.crop_size
        d1, h1, w1 = pos1
        d2, h2, w2 = pos2

        # Cached files are always saved as 3D (D, H, W)
        crop1 = np.array(vol[d1:d1+c, h1:h1+c, w1:w1+c], dtype=np.float32)
        crop2 = np.array(vol[d2:d2+c, h2:h2+c, w2:w2+c], dtype=np.float32)
        return crop1, crop2

    def _cache_full_volume(self, path: Path, cache_path: Path) -> None:
        """Decompress full volume and save to local SSD cache.

        Called once per file during the first epoch. Subsequent epochs
        read from cache via mmap.
        """
        try:
            if path.suffix == '.b2nd' and HAS_BLOSC2:
                arr = blosc2.open(str(path), mode='r')
                vol = np.asarray(arr[:], dtype=np.float32)
            else:
                vol = np.load(str(path)).astype(np.float32)

            # Squeeze to 3D for consistent cache format
            while vol.ndim > 3 and vol.shape[0] == 1:
                vol = vol.squeeze(0)

            np.save(str(cache_path), vol)
        except (OSError, Exception):
            pass  # cache full, permissions, etc. — just skip

    def _read_full_and_cache(
        self, path: Path, cache_path: Path,
        pos1: Tuple[int, ...], pos2: Tuple[int, ...]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Read full volume, extract crops, and save cache in one pass.

        On the first epoch when caching is enabled, this does 1 full read
        instead of 2 partial reads + 1 full read for caching (3x I/O).
        """
        arr = blosc2.open(str(path), mode='r')
        vol = np.asarray(arr[:], dtype=np.float32)

        # Squeeze to 3D
        while vol.ndim > 3 and vol.shape[0] == 1:
            vol = vol.squeeze(0)

        # Extract crops from the in-memory volume
        c = self.crop_size
        d1, h1, w1 = pos1
        d2, h2, w2 = pos2
        crop1 = vol[d1:d1+c, h1:h1+c, w1:w1+c].copy()
        crop2 = vol[d2:d2+c, h2:h2+c, w2:w2+c].copy()

        # Save full volume to cache (async-ish: fire and forget)
        try:
            np.save(str(cache_path), vol)
        except (OSError, Exception):
            pass  # disk full, permissions — skip silently

        return crop1, crop2

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a volume and return two augmented crop views.

        Strategy:
          1. Check local cache → mmap + slice (fast, ~ms)
          2. Else: blosc2 partial read from NFS (slower, ~seconds)
             Also triggers cache save of full volume for next epoch.

        Returns:
            dict with 'view1', 'view2' (each [1, D, H, W]), and 'index'
        """
        try:
            return self._load_sample(idx)
        except Exception as e:
            # Fallback: pick a random different sample (matches nnssl behavior)
            fallback_idx = int(np.random.randint(0, len(self)))
            try:
                return self._load_sample(fallback_idx)
            except Exception:
                # Last resort: return zeros (should be extremely rare)
                c = self.crop_size
                zeros = torch.zeros(1, c, c, c)
                return {'view1': zeros, 'view2': zeros, 'index': idx}

    def _load_sample(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.file_paths[idx]
        cache_path = self._cache_path(path)

        # --- Determine spatial shape (metadata only, no decompression) ---
        if cache_path is not None and cache_path.exists():
            # Cached: read npy header for shape
            with open(cache_path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                raw_shape, _, _ = np.lib.format._read_array_header(f, version)
            spatial_shape = self._get_spatial_shape(raw_shape)
            use_cache = True
        elif path.suffix == '.b2nd' and HAS_BLOSC2:
            arr_handle = blosc2.open(str(path), mode='r')
            spatial_shape = self._get_spatial_shape(arr_handle.shape)
            use_cache = False
        elif path.suffix == '.npy':
            with open(path, 'rb') as f:
                version = np.lib.format.read_magic(f)
                raw_shape, _, _ = np.lib.format._read_array_header(f, version)
            spatial_shape = self._get_spatial_shape(raw_shape)
            use_cache = False
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # --- Compute crop positions (data-free) ---
        pos1, pos2 = compute_crop_positions(
            spatial_shape, self.crop_size, self.min_overlap)

        # --- Read crops ---
        if use_cache:
            crop1, crop2 = self._read_crops_cached(cache_path, pos1, pos2)
        elif path.suffix == '.b2nd' and HAS_BLOSC2:
            if cache_path is not None and not cache_path.exists():
                # First epoch with caching: read FULL volume once, extract
                # crops, then save cache. This is 1 full read instead of
                # 2 partial reads + 1 full read for caching.
                crop1, crop2 = self._read_full_and_cache(path, cache_path, pos1, pos2)
            else:
                # No caching: partial reads only (fast, ~11% of volume)
                crop1, crop2 = self._read_crops_blosc2(path, pos1, pos2)
        else:
            # .npy fallback: load full and slice
            vol = np.load(str(path)).astype(np.float32)
            while vol.ndim > 3 and vol.shape[0] == 1:
                vol = vol.squeeze(0)
            c = self.crop_size
            d1, h1, w1 = pos1
            d2, h2, w2 = pos2
            crop1 = vol[d1:d1+c, h1:h1+c, w1:w1+c]
            crop2 = vol[d2:d2+c, h2:h2+c, w2:w2+c]

        # --- To tensor: [1, D, H, W] ---
        crop1_t = torch.from_numpy(crop1).float().unsqueeze(0)
        crop2_t = torch.from_numpy(crop2).float().unsqueeze(0)

        # --- Apply augmentations ---
        # Augment3D expects [B, C, D, H, W], so add and remove batch dim
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
        crop1 = torch.from_numpy(vol[d1:d1+c, h1:h1+c, w1:w1+c]).float().unsqueeze(0)
        crop2 = torch.from_numpy(vol[d2:d2+c, h2:h2+c, w2:w2+c]).float().unsqueeze(0)

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
    """Create a DataLoader for BYOL pretraining.

    If data_dir is provided, loads real blosc2/npy data.
    If synthetic=True, uses SyntheticBYOLDataset for testing.
    """
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
        persistent_workers=num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=_worker_init_fn,
    )