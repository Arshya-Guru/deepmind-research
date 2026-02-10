"""
Smoke tests for networks_3d.py

Uses tiny 16^3 volumes with batch=2 to fit in sandbox memory.
Tests shape flow, gradient flow, EMA update, loss computation,
and the full BYOL <-> segmentation pipeline.
"""
import sys
sys.path.insert(0, '/home/claude')

import torch
import torch.nn.functional as F

from byol3d.utils.networks_3d import (
    ConvBlock3D,
    ResConvBlock3D,
    UNetEncoder3D,
    ProjectionMLP,
    PredictionMLP,
    BYOLNetwork,
    BYOLTargetNetwork,
    regression_loss,
    update_target_ema,
    cosine_ema_schedule,
    create_byol_pair,
    UNetDecoder3D,
    SegmentationUNet3D,
)


# ---- Test config: tiny to avoid OOM ----
B, C, S = 2, 1, 16  # batch=2, channels=1, spatial=16^3
NUM_LEVELS = 3       # only 3 levels for 16^3 (16->8->4 bottleneck)
BASE_CH = 16         # tiny channels
PROJ_HIDDEN = 64
PROJ_OUT = 32
PRED_HIDDEN = 64

torch.manual_seed(42)


def test_conv_blocks():
    x = torch.randn(B, 8, S, S, S)

    plain = ConvBlock3D(8, 16)
    y = plain(x)
    assert y.shape == (B, 16, S, S, S), f"ConvBlock3D shape: {y.shape}"

    res = ResConvBlock3D(8, 16)
    y = res(x)
    assert y.shape == (B, 16, S, S, S), f"ResConvBlock3D shape: {y.shape}"

    res_same = ResConvBlock3D(8, 8)
    y = res_same(x)
    assert y.shape == (B, 8, S, S, S)

    print("[PASS] conv blocks")


def test_encoder_gap_mode():
    enc = UNetEncoder3D(
        in_channels=C, base_channels=BASE_CH, num_levels=NUM_LEVELS
    )
    x = torch.randn(B, C, S, S, S)
    y = enc(x, return_skips=False)

    assert y.ndim == 2
    assert y.shape == (B, enc.repr_dim)
    print(f"[PASS] encoder GAP mode -> {y.shape}, repr_dim={enc.repr_dim}, channels={enc.channels}")


def test_encoder_skip_mode():
    enc = UNetEncoder3D(
        in_channels=C, base_channels=BASE_CH, num_levels=NUM_LEVELS
    )
    x = torch.randn(B, C, S, S, S)
    bottleneck, skips = enc(x, return_skips=True)

    assert len(skips) == NUM_LEVELS - 1
    expected_spatial = S
    for i, skip in enumerate(skips):
        assert skip.shape == (B, enc.channels[i], expected_spatial, expected_spatial, expected_spatial)
        expected_spatial //= 2

    assert bottleneck.shape[1] == enc.channels[-1]
    assert bottleneck.shape[2] == expected_spatial
    print(f"[PASS] encoder skip mode -> bottleneck {bottleneck.shape}, "
          f"skips: {[s.shape for s in skips]}")


def test_mlp_heads():
    repr_dim = 64
    x = torch.randn(B, repr_dim)

    proj = ProjectionMLP(repr_dim, PROJ_HIDDEN, PROJ_OUT)
    z = proj(x)
    assert z.shape == (B, PROJ_OUT)

    pred = PredictionMLP(PROJ_OUT, PRED_HIDDEN, PROJ_OUT)
    q = pred(z)
    assert q.shape == (B, PROJ_OUT)

    print(f"[PASS] MLP heads: proj {x.shape}->{z.shape}, pred {z.shape}->{q.shape}")


def test_regression_loss():
    x = torch.randn(B, PROJ_OUT)
    y = torch.randn(B, PROJ_OUT)

    loss = regression_loss(x, y)
    assert loss.shape == (B,)
    assert (loss >= 0).all() and (loss <= 4).all()

    loss_same = regression_loss(x, x)
    assert loss_same.abs().max() < 1e-5

    loss_opp = regression_loss(x, -x)
    assert (loss_opp - 4.0).abs().max() < 1e-5

    print(f"[PASS] regression_loss: random={loss.mean():.4f}, same={loss_same.mean():.6f}, "
          f"opposite={loss_opp.mean():.4f}")


def test_ema_schedule():
    tau_0 = cosine_ema_schedule(0, base_ema=0.99, max_steps=1000)
    tau_mid = cosine_ema_schedule(500, base_ema=0.99, max_steps=1000)
    tau_end = cosine_ema_schedule(1000, base_ema=0.99, max_steps=1000)

    assert abs(tau_0 - 0.99) < 1e-6
    assert abs(tau_end - 1.0) < 1e-6
    assert tau_0 < tau_mid < tau_end

    print(f"[PASS] EMA schedule: tau(0)={tau_0:.4f}, tau(500)={tau_mid:.4f}, tau(1000)={tau_end:.4f}")


def test_byol_pair_creation():
    online, target = create_byol_pair(
        in_channels=C, base_channels=BASE_CH, num_levels=NUM_LEVELS,
        projector_hidden_dim=PROJ_HIDDEN, projector_output_dim=PROJ_OUT,
        predictor_hidden_dim=PRED_HIDDEN,
    )

    for p in target.parameters():
        assert not p.requires_grad

    for p_o, p_t in zip(online.encoder.parameters(), target.encoder.parameters()):
        assert torch.equal(p_o.data, p_t.data)

    for p_o, p_t in zip(online.projector.parameters(), target.projector.parameters()):
        assert torch.equal(p_o.data, p_t.data)

    online_params = sum(p.numel() for p in online.parameters())
    target_params = sum(p.numel() for p in target.parameters())
    print(f"[PASS] BYOL pair: online={online_params:,} params, target={target_params:,} params")

    return online, target


def test_byol_forward_and_loss(online, target):
    x = torch.randn(B, C, S, S, S)
    view1 = x + 0.1 * torch.randn_like(x)
    view2 = x + 0.1 * torch.randn_like(x)

    online.train()
    out1 = online(view1)
    out2 = online(view2)

    target.eval()
    with torch.no_grad():
        tgt1 = target(view1)
        tgt2 = target(view2)

    # Symmetrized BYOL loss (paper Eq. 2)
    loss = regression_loss(out1['prediction'], tgt2['projection'].detach()).mean()
    loss = loss + regression_loss(out2['prediction'], tgt1['projection'].detach()).mean()

    assert loss.requires_grad
    print(f"[PASS] BYOL forward + loss = {loss.item():.4f}")

    return loss


def test_gradient_flow(online, loss):
    loss.backward()

    has_grad = False
    for name, p in online.named_parameters():
        if p.grad is not None and p.grad.abs().sum() > 0:
            has_grad = True
            break
    assert has_grad

    print("[PASS] gradient flow through online network")


def test_ema_update(online, target):
    target_param_before = target.encoder.encoders[0].conv1[0].weight.data.clone()

    with torch.no_grad():
        for p in online.parameters():
            p.data.add_(0.01 * torch.randn_like(p.data))

    update_target_ema(online, target, tau=0.99)

    target_param_after = target.encoder.encoders[0].conv1[0].weight.data
    diff = (target_param_after - target_param_before).abs().sum()
    assert diff > 0

    print(f"[PASS] EMA update: param diff = {diff.item():.6f}")


def test_segmentation_unet():
    enc = UNetEncoder3D(
        in_channels=C, base_channels=BASE_CH, num_levels=NUM_LEVELS
    )
    seg = SegmentationUNet3D(encoder=enc, num_classes=1)

    x = torch.randn(B, C, S, S, S)
    logits = seg(x)
    assert logits.shape == (B, 1, S, S, S)

    loss = logits.mean()
    loss.backward()
    enc_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in seg.encoder.parameters())
    dec_has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in seg.decoder.parameters())
    assert enc_has_grad
    assert dec_has_grad

    seg_params = sum(p.numel() for p in seg.parameters())
    print(f"[PASS] segmentation U-Net: input {x.shape} -> logits {logits.shape}, "
          f"{seg_params:,} params")


def test_from_byol_encoder():
    online, _ = create_byol_pair(
        in_channels=C, base_channels=BASE_CH, num_levels=NUM_LEVELS,
        projector_hidden_dim=PROJ_HIDDEN, projector_output_dim=PROJ_OUT,
        predictor_hidden_dim=PRED_HIDDEN,
    )

    with torch.no_grad():
        for p in online.encoder.parameters():
            p.data.add_(torch.randn_like(p.data))

    seg = SegmentationUNet3D.from_byol_encoder(
        online.encoder, num_classes=1, freeze_encoder=False
    )

    for p_byol, p_seg in zip(online.encoder.parameters(), seg.encoder.parameters()):
        assert torch.equal(p_byol.data, p_seg.data)

    x = torch.randn(B, C, S, S, S)
    logits = seg(x)
    assert logits.shape == (B, 1, S, S, S)

    seg_frozen = SegmentationUNet3D.from_byol_encoder(
        online.encoder, num_classes=1, freeze_encoder=True
    )
    for p in seg_frozen.encoder.parameters():
        assert not p.requires_grad
    for p in seg_frozen.decoder.parameters():
        assert p.requires_grad

    print("[PASS] from_byol_encoder: weights transferred, frozen/unfrozen modes work")


def test_full_size_shape_check():
    """Verify param counts at real 96^3 config (no forward pass)."""
    enc = UNetEncoder3D(in_channels=1, base_channels=32, num_levels=5)
    print(f"\n--- Real config (96^3 input, no forward pass) ---")
    print(f"  Encoder channels: {enc.channels}")
    print(f"  repr_dim: {enc.repr_dim}")
    s = 96
    sizes = []
    for i in range(enc.num_levels):
        sizes.append(f"L{i}:{s}^3 ({enc.channels[i]}ch)")
        if i < enc.num_levels - 1:
            s //= 2
    print(f"  Levels: {'  '.join(sizes)}")
    print(f"  After GAP: [{enc.repr_dim}]-dim vector")

    online, target = create_byol_pair(
        in_channels=1, base_channels=32, num_levels=5,
        projector_hidden_dim=2048, projector_output_dim=256,
        predictor_hidden_dim=2048,
    )
    n_online = sum(p.numel() for p in online.parameters())
    n_target = sum(p.numel() for p in target.parameters())
    seg = SegmentationUNet3D(
        encoder=UNetEncoder3D(in_channels=1, base_channels=32, num_levels=5),
        num_classes=1,
    )
    n_seg = sum(p.numel() for p in seg.parameters())
    print(f"\n  Param counts (real config):")
    print(f"    Online network:  {n_online:>12,}")
    print(f"    Target network:  {n_target:>12,}")
    print(f"    Seg U-Net:       {n_seg:>12,}")
    print(f"[PASS] real config shape trace")


if __name__ == '__main__':
    print("=" * 60)
    print(f"Smoke test: B={B}, spatial={S}^3, levels={NUM_LEVELS}, base_ch={BASE_CH}")
    print("=" * 60)

    test_conv_blocks()
    test_encoder_gap_mode()
    test_encoder_skip_mode()
    test_mlp_heads()
    test_regression_loss()
    test_ema_schedule()

    online, target = test_byol_pair_creation()
    loss = test_byol_forward_and_loss(online, target)
    test_gradient_flow(online, loss)
    test_ema_update(online, target)

    test_segmentation_unet()
    test_from_byol_encoder()
    test_full_size_shape_check()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
