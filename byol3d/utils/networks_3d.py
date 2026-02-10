"""
3D network components for BYOL pretraining and segmentation fine-tuning.

Architecture follows the BYOL paper (Grill et al., 2020) adapted for 3D:
  - Encoder f_θ: contracting path of a 3D U-Net (stores skip connections)
  - Projector g_θ: 2-layer MLP with BN, matching paper's depth-2 default
  - Predictor q_θ: same architecture as projector (paper Section 3.3)
  - SegmentationDecoder: expanding path with skip connections for fine-tuning

Key 2D→3D adaptations:
  - All Conv2D → Conv3D, all pool ops → 3D
  - InstanceNorm3d in encoder (not BatchNorm3d) — batch sizes of 2-8 make
    3D BN statistics unreliable; InstanceNorm is standard in nnU-Net
  - BatchNorm1d kept in projector/predictor MLPs — paper shows this is
    critical for preventing collapse, and 1D BN is stable even at batch=2
  - LeakyReLU in encoder (nnU-Net convention), ReLU in MLPs (paper convention)

Notation follows the paper's Figure 2:
  x → t(x)=v → f_θ(v)=y_θ → g_θ(y_θ)=z_θ → q_θ(z_θ) → loss with sg(z'_ξ)
"""

from __future__ import annotations

import copy
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class ConvBlock3D(nn.Module):
    """Two 3x3x3 convolutions with InstanceNorm + LeakyReLU.

    Standard U-Net double-conv block adapted for 3D.
    InstanceNorm chosen over BatchNorm for robustness at small batch sizes.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResConvBlock3D(nn.Module):
    """Residual variant: adds a skip from input to output.

    If in_channels != out_channels, a 1x1x1 conv projects the residual.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
        )
        self.skip = (
            nn.Conv3d(in_channels, out_channels, 1, bias=False)
            if in_channels != out_channels
            else nn.Identity()
        )
        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + self.skip(x))


# ---------------------------------------------------------------------------
# Encoder  (f_theta in paper)
# ---------------------------------------------------------------------------

class UNetEncoder3D(nn.Module):
    """Contracting path of a 3D U-Net.

    During BYOL pretraining:
      bottleneck -> AdaptiveAvgPool3d(1) -> flatten -> projector -> predictor

    During segmentation fine-tuning:
      skip connections at each level feed the decoder.

    Default channels [32, 64, 128, 256, 320] match nnU-Net's "3d_fullres"
    configuration for patch sizes around 96^3-128^3.  The bottleneck (320 ch
    at 6x6x6 for a 96^3 input) gives a 320-dim representation after GAP.

    Args:
        in_channels: number of input channels (1 for single-channel lightsheet).
        base_channels: number of channels after the first conv block.
        num_levels: depth of the encoder (number of downsampling steps + 1).
        channels: explicit channel counts per level.  If None, doubles from
            base_channels up to a cap of 320 (nnU-Net style).
        use_residuals: if True, use ResConvBlock3D instead of ConvBlock3D.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        num_levels: int = 5,
        channels: Optional[Sequence[int]] = None,
        use_residuals: bool = True,
    ):
        super().__init__()
        if channels is None:
            channels = []
            c = base_channels
            for i in range(num_levels):
                channels.append(min(c, 320))
                c *= 2
        self.channels = list(channels)
        self.num_levels = num_levels

        Block = ResConvBlock3D if use_residuals else ConvBlock3D

        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()

        for i in range(num_levels):
            c_in = in_channels if i == 0 else self.channels[i - 1]
            c_out = self.channels[i]
            self.encoders.append(Block(c_in, c_out))
            if i < num_levels - 1:
                self.pools.append(nn.MaxPool3d(kernel_size=2, stride=2))

        # global average pool for BYOL head -- produces [B, C_bottleneck]
        self.gap = nn.AdaptiveAvgPool3d(1)

        # representation dimensionality (y_theta in the paper)
        self.repr_dim = self.channels[-1]

    def forward(
        self, x: torch.Tensor, return_skips: bool = False
    ):
        """
        Args:
            x: [B, C_in, D, H, W]
            return_skips: if True, also return skip connections for the decoder.

        Returns:
            If return_skips is False:
                representation y_theta of shape [B, repr_dim]  (after GAP+flatten)
            If return_skips is True:
                (bottleneck_features [B, C_bot, D', H', W'],
                 skips list from high-res to low-res)
        """
        skips = []
        out = x
        for i in range(self.num_levels):
            out = self.encoders[i](out)
            if i < self.num_levels - 1:
                skips.append(out)
                out = self.pools[i](out)

        if return_skips:
            # for segmentation: return full bottleneck volume + skips
            return out, skips

        # for BYOL: global average pool -> [B, repr_dim]
        return self.gap(out).flatten(1)


# ---------------------------------------------------------------------------
# Projector g_theta  and  Predictor q_theta  (MLP heads from paper)
# ---------------------------------------------------------------------------

class ProjectionMLP(nn.Module):
    """2-layer MLP: Linear -> BN -> ReLU -> Linear.

    Matches the paper's default architecture (Table 14, depth=2 optimal).
    Paper dims: hidden=4096, output=256 for ResNet-50 (repr_dim=2048).
    We scale down since our repr_dim is 320, not 2048.

    Note: final linear has no bias (matching paper's Haiku code) and no BN
    on the output (paper: "Contrary to SimCLR, the output of this MLP is
    not batch normalized").
    """

    def __init__(self, input_dim: int, hidden_dim: int = 2048, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PredictionMLP(nn.Module):
    """Predictor q_theta -- same architecture as projector.

    Paper: "The predictor q_theta uses the same architecture as g_theta."
    Input dim = projector output dim, hidden can differ.
    """

    def __init__(self, input_dim: int = 256, hidden_dim: int = 2048, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Full BYOL online/target network wrapper
# ---------------------------------------------------------------------------

def regression_loss(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """BYOL regression loss: L2 distance between L2-normalized vectors.

    Equivalent to 2 - 2*cosine_similarity (paper Eq. 2).
    Mirrors helpers.regression_loss from the original repo.
    """
    x = F.normalize(x, dim=-1)
    y = F.normalize(y, dim=-1)
    return 2 - 2 * (x * y).sum(dim=-1)


class BYOLNetwork(nn.Module):
    """Complete BYOL architecture wrapping encoder + projector + predictor.

    Used for the *online* network.  The target network is a separate copy
    whose parameters are updated via EMA (handled externally).

    Forward returns dict with 'projection' and 'prediction' keys,
    matching the original repo's forward output structure.
    """

    def __init__(
        self,
        encoder: UNetEncoder3D,
        projector_hidden_dim: int = 2048,
        projector_output_dim: int = 256,
        predictor_hidden_dim: int = 2048,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionMLP(
            input_dim=encoder.repr_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
        )
        self.predictor = PredictionMLP(
            input_dim=projector_output_dim,
            hidden_dim=predictor_hidden_dim,
            output_dim=projector_output_dim,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = self.encoder(x, return_skips=False)     # representation
        z = self.projector(y)                         # projection
        q = self.predictor(z)                         # prediction
        return {'projection': z, 'prediction': q}


class BYOLTargetNetwork(nn.Module):
    """Target network: encoder + projector only (no predictor).

    Paper: "Note that this predictor is only applied to the online branch,
    making the architecture asymmetric."
    """

    def __init__(
        self,
        encoder: UNetEncoder3D,
        projector_hidden_dim: int = 2048,
        projector_output_dim: int = 256,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionMLP(
            input_dim=encoder.repr_dim,
            hidden_dim=projector_hidden_dim,
            output_dim=projector_output_dim,
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        y = self.encoder(x, return_skips=False)
        z = self.projector(y)
        return {'projection': z}


# ---------------------------------------------------------------------------
# EMA helpers  (mirrors schedules.py target_ema + tree update)
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_target_ema(
    online: nn.Module,
    target: nn.Module,
    tau: float,
) -> None:
    """Exponential moving average update: xi <- tau*xi + (1-tau)*theta.

    Matches the original repo's:
        target_params = tree_map(lam x,y: x + (1-tau)(y-x), target, online)
    which simplifies to tau*target + (1-tau)*online.
    """
    for p_online, p_target in zip(online.parameters(), target.parameters()):
        p_target.data.mul_(tau).add_(p_online.data, alpha=1.0 - tau)


def cosine_ema_schedule(step: int, base_ema: float, max_steps: int) -> float:
    """Cosine EMA schedule from paper: tau = 1 - (1-tau_base)*(cos(pi*k/K)+1)/2.

    Mirrors schedules.target_ema from the original repo.
    At step 0: tau = base_ema.  At step max_steps: tau -> 1.0.
    """
    progress = min(step / max(max_steps, 1), 1.0)
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return 1.0 - (1.0 - base_ema) * decay


# ---------------------------------------------------------------------------
# BYOL factory -- creates matched online + target pairs
# ---------------------------------------------------------------------------

def create_byol_pair(
    in_channels: int = 1,
    base_channels: int = 32,
    num_levels: int = 5,
    encoder_channels: Optional[Sequence[int]] = None,
    use_residuals: bool = True,
    projector_hidden_dim: int = 2048,
    projector_output_dim: int = 256,
    predictor_hidden_dim: int = 2048,
) -> Tuple[BYOLNetwork, BYOLTargetNetwork]:
    """Create online and target networks with shared initial weights.

    Paper: "Both online and target networks share the same architecture."
    Target is initialized as a copy of online (minus the predictor).
    """
    encoder = UNetEncoder3D(
        in_channels=in_channels,
        base_channels=base_channels,
        num_levels=num_levels,
        channels=encoder_channels,
        use_residuals=use_residuals,
    )
    online = BYOLNetwork(
        encoder=encoder,
        projector_hidden_dim=projector_hidden_dim,
        projector_output_dim=projector_output_dim,
        predictor_hidden_dim=predictor_hidden_dim,
    )

    # Build target with its own encoder + projector (no predictor)
    target_encoder = copy.deepcopy(encoder)
    target = BYOLTargetNetwork(
        encoder=target_encoder,
        projector_hidden_dim=projector_hidden_dim,
        projector_output_dim=projector_output_dim,
    )
    # Initialize target projector from online projector
    target.projector.load_state_dict(online.projector.state_dict())

    # Target parameters are not updated by gradient -- freeze them
    for p in target.parameters():
        p.requires_grad = False

    return online, target


# ---------------------------------------------------------------------------
# Segmentation decoder  (for fine-tuning phase)
# ---------------------------------------------------------------------------

class UNetDecoder3D(nn.Module):
    """Expanding path of a 3D U-Net for segmentation fine-tuning.

    Takes bottleneck features + skip connections from the encoder and
    produces a dense prediction at input resolution.
    """

    def __init__(
        self,
        encoder_channels: Sequence[int],
        num_classes: int = 1,
        use_residuals: bool = True,
    ):
        super().__init__()
        self.num_classes = num_classes
        Block = ResConvBlock3D if use_residuals else ConvBlock3D

        dec_channels = list(reversed(encoder_channels[:-1]))  # [256, 128, 64, 32]
        bottleneck_ch = encoder_channels[-1]  # 320

        self.upconvs = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()

        in_ch = bottleneck_ch
        for out_ch in dec_channels:
            self.upconvs.append(
                nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
            )
            # After concat with skip: out_ch (from upconv) + out_ch (from skip)
            self.dec_blocks.append(Block(out_ch * 2, out_ch))
            in_ch = out_ch

        self.final_conv = nn.Conv3d(dec_channels[-1], num_classes, kernel_size=1)

    def forward(
        self,
        bottleneck: torch.Tensor,
        skips: List[torch.Tensor],
    ) -> torch.Tensor:
        """
        Args:
            bottleneck: [B, C_bot, D', H', W'] from encoder
            skips: list from HIGH-res to LOW-res (encoder output order)

        Returns:
            logits: [B, num_classes, D, H, W] at input resolution
        """
        # skips are high->low, decoder needs low->high
        skips = list(reversed(skips))
        x = bottleneck
        for i, (upconv, dec_block) in enumerate(zip(self.upconvs, self.dec_blocks)):
            x = upconv(x)
            skip = skips[i]
            # handle size mismatches from non-power-of-2 inputs
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='trilinear',
                                  align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x = dec_block(x)

        return self.final_conv(x)


class SegmentationUNet3D(nn.Module):
    """Full 3D U-Net for segmentation fine-tuning.

    Loads pretrained encoder weights from BYOL, attaches a fresh decoder.

    Usage:
        # After BYOL pretraining:
        seg_model = SegmentationUNet3D.from_byol_encoder(
            online_network.encoder, num_classes=1)
        # Or create fresh:
        seg_model = SegmentationUNet3D(encoder, num_classes=1)
    """

    def __init__(
        self,
        encoder: UNetEncoder3D,
        num_classes: int = 1,
        use_residuals: bool = True,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = UNetDecoder3D(
            encoder_channels=encoder.channels,
            num_classes=num_classes,
            use_residuals=use_residuals,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns logits [B, num_classes, D, H, W]."""
        bottleneck, skips = self.encoder(x, return_skips=True)
        return self.decoder(bottleneck, skips)

    @classmethod
    def from_byol_encoder(
        cls,
        byol_encoder: UNetEncoder3D,
        num_classes: int = 1,
        use_residuals: bool = True,
        freeze_encoder: bool = False,
    ) -> 'SegmentationUNet3D':
        """Create segmentation model with pretrained BYOL encoder weights.

        Args:
            byol_encoder: encoder from the trained online network.
            num_classes: number of segmentation classes (1 for binary Abeta).
            use_residuals: whether decoder uses residual blocks.
            freeze_encoder: if True, encoder params are frozen (linear probe).
        """
        encoder = copy.deepcopy(byol_encoder)

        if freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False

        return cls(encoder=encoder, num_classes=num_classes,
                   use_residuals=use_residuals)
