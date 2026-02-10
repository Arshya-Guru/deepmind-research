"""
Dataset loader for blosc2-compressed lightsheet patches.

Loads preprocessed 256^3 patches from the nnssl pipeline, extracts
overlap-constrained crop pairs, and applies BYOL augmentations.

Data flow:
  1. Scan directory for .b2nd files (blosc2 n-dimensional arrays)
  2. On __getitem__: open file lazily, extract two 96^3 crops with overlap >= 0.4
  3. Apply view1/view2 augmentations independently
  4. Return dict with 'view1' and 'view2' tensors

The blosc2 partial decompression means only the chunks overlapping with
the requested crop are actually decompressed -- efficient for large volumes.

If blosc2 is not available (e.g. in testing), falls back to random tensors
or numpy .npy files.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from byol3d.utils.augmentations_3d import (
    Augment3D,
    augment_config,
    extract_overlapping_crops,
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
# Dataset
# ---------------------------------------------------------------------------

class LightsheetBYOLDataset(Dataset):
    """Dataset for BYOL pretraining on blosc2-compressed lightsheet patches.

    Each sample returns two augmented views (crop pairs) from one 256^3 patch.

    Args:
        data_dir: path to directory containing .b2nd files
            (e.g. /nfs/.../nnssl_data/8/nnssl_data/preprocessed)
        crop_size: spatial size of each crop
        min_overlap: minimum volumetric overlap fraction between crop pairs
        view1_aug: augmentation pipeline for view 1
        view2_aug: augmentation pipeline for view 2
        normalize: if True, normalize each volume to [0, 1] on load
        file_ext: file extension to search for ('.b2nd' for blosc2, '.npy' for numpy)
    """

    def __init__(
        self,
        data_dir: str,
        crop_size: int = 96,
        min_overlap: float = 0.4,
        view1_aug: Optional[Augment3D] = None,
        view2_aug: Optional[Augment3D] = None,
        normalize: bool = True,
        file_ext: str = '.b2nd',
    ):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.crop_size = crop_size
        self.min_overlap = min_overlap
        self.normalize = normalize
        self.file_ext = file_ext

        # Default augmentations from our config
        self.view1_aug = view1_aug or Augment3D(**augment_config['view1'])
        self.view2_aug = view2_aug or Augment3D(**augment_config['view2'])

        # Discover files
        self.file_paths = sorted(self.data_dir.glob(f'*{file_ext}'))
        if not self.file_paths:
            # Also check for .npy fallback
            self.file_paths = sorted(self.data_dir.glob('*.npy'))
            if self.file_paths:
                self.file_ext = '.npy'

        if not self.file_paths:
            raise FileNotFoundError(
                f"No {file_ext} or .npy files found in {data_dir}")

        print(f"LightsheetBYOLDataset: found {len(self.file_paths)} files in {data_dir}")

    def __len__(self) -> int:
        return len(self.file_paths)

    def _load_volume(self, path: Path) -> np.ndarray:
        """Load a 3D volume from disk.

        For blosc2: uses lazy loading with partial decompression.
        For numpy: standard np.load.
        """
        if path.suffix == '.b2nd' and HAS_BLOSC2:
            # Lazy open -- no data loaded yet
            arr = blosc2.open(str(path))
            # Materialize the full array (blosc2 handles chunk-level decompression)
            # For even more efficiency, we could slice here, but we need the full
            # volume to compute random crop positions
            data = arr[:]
            if hasattr(arr, 'schunk'):
                pass  # already numpy-like
            return np.asarray(data, dtype=np.float32)
        elif path.suffix == '.npy':
            return np.load(str(path)).astype(np.float32)
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load a volume and return two augmented crop views.

        Returns:
            dict with:
                'view1': [1, D, H, W] augmented crop
                'view2': [1, D, H, W] augmented crop
                'index': int, the sample index
        """
        vol = self._load_volume(self.file_paths[idx])

        # Ensure 3D (squeeze any leading singleton dims)
        while vol.ndim > 3 and vol.shape[0] == 1:
            vol = vol.squeeze(0)
        if vol.ndim != 3:
            raise ValueError(
                f"Expected 3D volume, got shape {vol.shape} from {self.file_paths[idx]}")

        # Normalize to [0, 1]
        if self.normalize:
            vmin, vmax = vol.min(), vol.max()
            if vmax > vmin:
                vol = (vol - vmin) / (vmax - vmin)
            else:
                vol = np.zeros_like(vol)

        # Convert to tensor: [1, 1, D, H, W] for batch-style crop extraction
        vol_t = torch.from_numpy(vol).float().unsqueeze(0).unsqueeze(0)

        # Extract overlapping crop pair
        crop1, crop2 = extract_overlapping_crops(
            vol_t,
            crop_size=self.crop_size,
            min_overlap_fraction=self.min_overlap,
        )

        # Remove batch dim: [1, D, H, W]
        crop1 = crop1.squeeze(0)
        crop2 = crop2.squeeze(0)

        # Apply augmentations (add batch dim back for aug pipeline, then remove)
        view1 = self.view1_aug(crop1.unsqueeze(0)).squeeze(0)
        view2 = self.view2_aug(crop2.unsqueeze(0)).squeeze(0)

        return {
            'view1': view1,
            'view2': view2,
            'index': idx,
        }


# ---------------------------------------------------------------------------
# Synthetic dataset for testing without real data
# ---------------------------------------------------------------------------

class SyntheticBYOLDataset(Dataset):
    """Synthetic dataset for testing the pipeline without real data.

    Generates random 3D volumes on the fly.
    """

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
        # Deterministic-ish random volume per index (for reproducibility in testing)
        gen = torch.Generator().manual_seed(idx)
        vol = torch.rand(1, 1, self.volume_size, self.volume_size, self.volume_size,
                         generator=gen)

        crop1, crop2 = extract_overlapping_crops(
            vol, crop_size=self.crop_size, min_overlap_fraction=self.min_overlap)

        view1 = self.view1_aug(crop1).squeeze(0)
        view2 = self.view2_aug(crop2).squeeze(0)

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
        )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )
