"""
3D augmentations for BYOL lightsheet pretraining.

Adapted from the paper's 2D augmentation pipeline (byol/utils/augmentations.py)
to 3D single-channel fluorescence microscopy.

Key differences from the original 2D pipeline:
  - No color jitter (single channel) -- replaced by intensity jitter + gamma
  - No solarization (physically nonsensical for fluorescence)
  - No grayscale conversion (already single channel)
  - Added: brightness gradient (simulates LSFM intensity falloff)
  - Added: Gaussian noise (simulates detector noise)
  - Gaussian blur sigma capped at 1.0 (not 2.0) to protect small plaques
  - All spatial ops are 3D

Augmentation table (from planning):
  | Augmentation        | Prob  | Parameters                          |
  |---------------------|-------|-------------------------------------|
  | Random crop         | 1.0   | 96^3 from 256^3, overlap >= 0.4    |
  | Random flip         | 0.5/ax| All 3 axes                          |
  | 90 deg rotation     | 0.5   | Isotropic data                      |
  | Intensity jitter    | 0.8   | brightness +/-0.1, contrast 0.75-1.25|
  | Gamma correction    | 0.5   | range (0.7, 1.5)                    |
  | Gaussian blur       | v1:0.5, v2:0.1 | sigma [0.1, 1.0]          |
  | Brightness gradient | 0.2   | range (-0.3, 0.3)                   |
  | Gaussian noise      | 0.3   | var (0, 0.03)                       |
  | Solarization        | 0.0   | REMOVED                             |

Usage:
    from byol3d.utils.augmentations_3d import Augment3D, augment_config

    aug_v1 = Augment3D(**augment_config['view1'])
    aug_v2 = Augment3D(**augment_config['view2'])

    view1 = aug_v1(crop1)  # crop1: [B, 1, D, H, W] float in [0, 1]
    view2 = aug_v2(crop2)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Default augmentation configs for view1 and view2
# ---------------------------------------------------------------------------

augment_config = dict(
    view1=dict(
        random_flip=True,
        random_rot90=True,
        intensity_jitter=dict(
            apply_prob=0.8,
            brightness_range=0.1,
            contrast_range=(0.75, 1.25),
        ),
        gamma_correction=dict(
            apply_prob=0.5,
            gamma_range=(0.7, 1.5),
        ),
        gaussian_blur=dict(
            apply_prob=0.5,
            sigma_min=0.1,
            sigma_max=1.0,
            kernel_size=5,
        ),
        brightness_gradient=dict(
            apply_prob=0.2,
            strength_range=(-0.3, 0.3),
        ),
        gaussian_noise=dict(
            apply_prob=0.3,
            var_range=(0.0, 0.03),
        ),
    ),
    view2=dict(
        random_flip=True,
        random_rot90=True,
        intensity_jitter=dict(
            apply_prob=0.8,
            brightness_range=0.1,
            contrast_range=(0.75, 1.25),
        ),
        gamma_correction=dict(
            apply_prob=0.5,
            gamma_range=(0.7, 1.5),
        ),
        gaussian_blur=dict(
            apply_prob=0.1,  # asymmetric: view2 has lower blur prob
            sigma_min=0.1,
            sigma_max=1.0,
            kernel_size=5,
        ),
        brightness_gradient=dict(
            apply_prob=0.2,
            strength_range=(-0.3, 0.3),
        ),
        gaussian_noise=dict(
            apply_prob=0.3,
            var_range=(0.0, 0.03),
        ),
    ),
)


# ---------------------------------------------------------------------------
# Individual augmentation functions (operate on [B, 1, D, H, W] tensors)
# ---------------------------------------------------------------------------

def random_flip_3d(x: torch.Tensor) -> torch.Tensor:
    """Random flip along each spatial axis with p=0.5 independently.

    Isotropic 3D data has no canonical orientation, so all axes are valid.
    """
    for dim in (2, 3, 4):  # D, H, W
        if torch.rand(1).item() < 0.5:
            x = torch.flip(x, [dim])
    return x


def random_rot90_3d(x: torch.Tensor) -> torch.Tensor:
    """Random 90-degree rotation around a random axis.

    For isotropic data, 90-degree rotations are exact (no interpolation needed).
    Picks one of 3 axis pairs and one of {0, 1, 2, 3} rotation counts.
    """
    if torch.rand(1).item() < 0.5:
        # pick a random plane to rotate in
        axis_pairs = [(2, 3), (2, 4), (3, 4)]  # (D,H), (D,W), (H,W)
        pair = axis_pairs[torch.randint(len(axis_pairs), (1,)).item()]
        k = torch.randint(1, 4, (1,)).item()  # 1, 2, or 3 rotations
        x = torch.rot90(x, k, pair)
    return x


def intensity_jitter(
    x: torch.Tensor,
    brightness_range: float = 0.1,
    contrast_range: Tuple[float, float] = (0.75, 1.25),
    apply_prob: float = 0.8,
) -> torch.Tensor:
    """Brightness and contrast jitter for single-channel volumes.

    Replaces the paper's color_transform for grayscale data.
    Narrower ranges than paper (0.4) to preserve biological signal.
    """
    if torch.rand(1).item() > apply_prob:
        return x

    # Brightness: additive shift
    brightness = (2 * torch.rand(1).item() - 1) * brightness_range
    x = x + brightness

    # Contrast: multiplicative scaling around the mean
    contrast = torch.empty(1).uniform_(*contrast_range).item()
    mean = x.mean()
    x = (x - mean) * contrast + mean

    return x.clamp(0, 1)


def gamma_correction(
    x: torch.Tensor,
    gamma_range: Tuple[float, float] = (0.7, 1.5),
    apply_prob: float = 0.5,
) -> torch.Tensor:
    """Random gamma correction.

    Replaces color augmentations for single-channel data.
    gamma < 1: brightens dark regions, gamma > 1: darkens them.
    """
    if torch.rand(1).item() > apply_prob:
        return x

    gamma = torch.empty(1).uniform_(*gamma_range).item()
    # Avoid issues with negative values
    x = x.clamp(min=1e-8)
    return x.pow(gamma).clamp(0, 1)


def gaussian_blur_3d(
    x: torch.Tensor,
    sigma_min: float = 0.1,
    sigma_max: float = 1.0,
    kernel_size: int = 5,
    apply_prob: float = 0.5,
) -> torch.Tensor:
    """3D Gaussian blur with random sigma.

    sigma capped at 1.0 (paper uses 2.0 for 224px images).
    At 4 um resolution, sigma=2.0 would blur ~10 um plaques.

    Uses separable 1D convolutions for efficiency (3 passes).
    """
    if torch.rand(1).item() > apply_prob:
        return x

    sigma = torch.empty(1).uniform_(sigma_min, sigma_max).item()

    # Build 1D Gaussian kernel
    half = kernel_size // 2
    coords = torch.arange(kernel_size, dtype=x.dtype, device=x.device) - half
    kernel_1d = torch.exp(-0.5 * (coords / max(sigma, 1e-8)) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()

    # Separable 3D blur: convolve along D, then H, then W
    # Reshape kernel for each axis and apply
    B, C_in, D, H, W = x.shape
    pad = half

    # Along D: kernel shape [1, 1, K, 1, 1]
    k_d = kernel_1d.view(1, 1, -1, 1, 1)
    x = F.pad(x, (0, 0, 0, 0, pad, pad), mode='replicate')
    x = F.conv3d(x, k_d.expand(C_in, -1, -1, -1, -1), groups=C_in)

    # Along H: kernel shape [1, 1, 1, K, 1]
    k_h = kernel_1d.view(1, 1, 1, -1, 1)
    x = F.pad(x, (0, 0, pad, pad, 0, 0), mode='replicate')
    x = F.conv3d(x, k_h.expand(C_in, -1, -1, -1, -1), groups=C_in)

    # Along W: kernel shape [1, 1, 1, 1, K]
    k_w = kernel_1d.view(1, 1, 1, 1, -1)
    x = F.pad(x, (pad, pad, 0, 0, 0, 0), mode='replicate')
    x = F.conv3d(x, k_w.expand(C_in, -1, -1, -1, -1), groups=C_in)

    return x


def brightness_gradient_3d(
    x: torch.Tensor,
    strength_range: Tuple[float, float] = (-0.3, 0.3),
    apply_prob: float = 0.2,
) -> torch.Tensor:
    """Additive linear brightness gradient along a random axis.

    Domain-specific augmentation simulating LSFM intensity falloff:
    as the light sheet penetrates tissue, signal attenuates.
    The gradient goes from -strength to +strength across the volume.
    """
    if torch.rand(1).item() > apply_prob:
        return x

    strength = torch.empty(1).uniform_(*strength_range).item()
    axis = torch.randint(0, 3, (1,)).item()  # 0=D, 1=H, 2=W

    spatial_dim = x.shape[axis + 2]  # +2 to skip B, C dims
    gradient = torch.linspace(-strength, strength, spatial_dim,
                              device=x.device, dtype=x.dtype)

    # Reshape to broadcast: e.g., for axis=0 (D): [1, 1, D, 1, 1]
    shape = [1, 1, 1, 1, 1]
    shape[axis + 2] = spatial_dim
    gradient = gradient.view(*shape)

    return (x + gradient).clamp(0, 1)


def gaussian_noise(
    x: torch.Tensor,
    var_range: Tuple[float, float] = (0.0, 0.03),
    apply_prob: float = 0.3,
) -> torch.Tensor:
    """Additive Gaussian noise simulating detector noise.

    Variance sampled uniformly from var_range.
    """
    if torch.rand(1).item() > apply_prob:
        return x

    var = torch.empty(1).uniform_(*var_range).item()
    std = var ** 0.5
    noise = torch.randn_like(x) * std
    return (x + noise).clamp(0, 1)


# ---------------------------------------------------------------------------
# Composed augmentation pipeline
# ---------------------------------------------------------------------------

class Augment3D:
    """Composed augmentation pipeline for one BYOL view.

    Applies augmentations in order: spatial (flip, rot90) -> intensity
    (jitter, gamma, blur, gradient, noise).

    Input:  [B, 1, D, H, W] float tensor in [0, 1]
    Output: [B, 1, D, H, W] float tensor in [0, 1]
    """

    def __init__(
        self,
        random_flip: bool = True,
        random_rot90: bool = True,
        intensity_jitter: Optional[dict] = None,
        gamma_correction: Optional[dict] = None,
        gaussian_blur: Optional[dict] = None,
        brightness_gradient: Optional[dict] = None,
        gaussian_noise: Optional[dict] = None,
    ):
        self.random_flip = random_flip
        self.random_rot90 = random_rot90
        self.intensity_jitter_cfg = intensity_jitter or {}
        self.gamma_correction_cfg = gamma_correction or {}
        self.gaussian_blur_cfg = gaussian_blur or {}
        self.brightness_gradient_cfg = brightness_gradient or {}
        self.gaussian_noise_cfg = gaussian_noise or {}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentation pipeline.

        Args:
            x: [B, 1, D, H, W] float tensor, values in [0, 1]

        Returns:
            Augmented tensor with same shape, values clamped to [0, 1]
        """
        # Spatial augmentations (applied per-sample for randomness)
        if self.random_flip:
            x = random_flip_3d(x)
        if self.random_rot90:
            x = random_rot90_3d(x)

        # Intensity augmentations
        if self.intensity_jitter_cfg:
            x = intensity_jitter(x, **self.intensity_jitter_cfg)
        if self.gamma_correction_cfg:
            x = gamma_correction(x, **self.gamma_correction_cfg)
        if self.gaussian_blur_cfg:
            x = gaussian_blur_3d(x, **self.gaussian_blur_cfg)
        if self.brightness_gradient_cfg:
            x = brightness_gradient_3d(x, **self.brightness_gradient_cfg)
        if self.gaussian_noise_cfg:
            x = gaussian_noise(x, **self.gaussian_noise_cfg)

        return x.clamp(0, 1)

    def __repr__(self) -> str:
        parts = [f"Augment3D(flip={self.random_flip}, rot90={self.random_rot90}"]
        if self.intensity_jitter_cfg:
            parts.append(f"  jitter={self.intensity_jitter_cfg}")
        if self.gamma_correction_cfg:
            parts.append(f"  gamma={self.gamma_correction_cfg}")
        if self.gaussian_blur_cfg:
            parts.append(f"  blur={self.gaussian_blur_cfg}")
        if self.brightness_gradient_cfg:
            parts.append(f"  gradient={self.brightness_gradient_cfg}")
        if self.gaussian_noise_cfg:
            parts.append(f"  noise={self.gaussian_noise_cfg}")
        return "\n".join(parts) + ")"


# ---------------------------------------------------------------------------
# Crop pair extraction with overlap constraint
# ---------------------------------------------------------------------------

def extract_overlapping_crops(
    volume: torch.Tensor,
    crop_size: int = 96,
    min_overlap_fraction: float = 0.4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract two crops from a volume with guaranteed minimum overlap.

    The overlap constraint ensures the two views share spatial content,
    so the BYOL loss encourages invariance to augmentation differences
    rather than requiring the model to hallucinate missing regions.

    For 96^3 crops from 256^3 patches, overlap >= 0.4 means at least
    ~40% of voxels are shared. Max offset per axis is ~25 voxels.

    Args:
        volume: [B, C, D, H, W] source volume (e.g. 256^3 patch)
        crop_size: spatial size of each crop (e.g. 96)
        min_overlap_fraction: minimum volumetric overlap between crops

    Returns:
        (crop1, crop2): each [B, C, crop_size, crop_size, crop_size]
    """
    _, _, D, H, W = volume.shape

    # Maximum start coordinate for any crop
    max_start_d = D - crop_size
    max_start_h = H - crop_size
    max_start_w = W - crop_size

    if max_start_d < 0 or max_start_h < 0 or max_start_w < 0:
        raise ValueError(
            f"Volume spatial dims ({D}, {H}, {W}) too small for crop_size={crop_size}")

    # First crop: random position
    d1 = torch.randint(0, max_start_d + 1, (1,)).item()
    h1 = torch.randint(0, max_start_h + 1, (1,)).item()
    w1 = torch.randint(0, max_start_w + 1, (1,)).item()

    # Second crop: constrained to overlap with first
    # Overlap along each axis = crop_size - |offset|
    # We need: product of overlaps / crop_size^3 >= min_overlap_fraction
    # Simplified: per-axis overlap >= min_overlap_fraction^(1/3)
    per_axis_min = min_overlap_fraction ** (1.0 / 3.0)
    max_offset = int(crop_size * (1.0 - per_axis_min))

    d2 = _constrained_start(d1, max_offset, max_start_d)
    h2 = _constrained_start(h1, max_offset, max_start_h)
    w2 = _constrained_start(w1, max_offset, max_start_w)

    crop1 = volume[:, :, d1:d1+crop_size, h1:h1+crop_size, w1:w1+crop_size]
    crop2 = volume[:, :, d2:d2+crop_size, h2:h2+crop_size, w2:w2+crop_size]

    return crop1.contiguous(), crop2.contiguous()


def _constrained_start(ref_start: int, max_offset: int, max_start: int) -> int:
    """Compute a random start position within max_offset of ref_start."""
    lo = max(0, ref_start - max_offset)
    hi = min(max_start, ref_start + max_offset)
    return torch.randint(lo, hi + 1, (1,)).item()
