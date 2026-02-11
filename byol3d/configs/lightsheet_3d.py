"""
Config for 3D lightsheet BYOL pretraining.

Mirrors configs/byol.py from the original repo, adapted for:
  - 3D U-Net encoder (not ResNet)
  - AdamW optimizer (not LARS — inappropriate for batch sizes << 256)
  - Single-channel fluorescence data
  - 96^3 crops from 256^3 patches

Preset EMA values follow the paper's Table 8:
  300 epochs: tau_base = 0.99
  1000 epochs: tau_base = 0.996
"""


# Presets keyed by number of training epochs
_EMA_PRESETS = {100: 0.99, 300: 0.99, 500: 0.993, 1000: 0.996}
_LR_PRESETS = {100: 1e-3, 300: 1e-3, 500: 5e-4, 1000: 3e-4}
_WD_PRESETS = {100: 1e-4, 300: 1e-4, 500: 1e-4, 1000: 1e-5}


def get_config(num_epochs: int = 300, batch_size: int = 4,
               num_samples: int = 1000):
    """Return config dict for 3D BYOL pretraining.

    Args:
        num_epochs: number of training epochs.
        batch_size: per-GPU batch size.
        num_samples: number of training volumes (patches).
    """
    steps_per_epoch = num_samples // batch_size
    max_steps = num_epochs * steps_per_epoch

    # Use nearest preset or default to 300-epoch values
    epoch_key = min(_EMA_PRESETS.keys(), key=lambda k: abs(k - num_epochs))

    config = dict(
        # ---- General ----
        random_seed=42,
        num_epochs=num_epochs,
        batch_size=batch_size,
        max_steps=max_steps,
        steps_per_epoch=steps_per_epoch,

        # ---- Data ----
        data=dict(
            # UPDATE THESE PATHS for your cluster
            data_dir='/nfs/khan/trainees/apooladi/abeta/nnssl_data/8/nnssl_data/preprocessed',
            crop_size=96,
            volume_size=256,   # source patch size
            min_overlap=0.4,
            num_workers=8,
            pin_memory=True,
            cache_dir='/tmp/byol_cache',   # local SSD cache for NFS volumes
            in_channels=1,
        ),

        # ---- Encoder (3D U-Net contracting path) ----
        encoder=dict(
            in_channels=1,
            base_channels=32,
            num_levels=5,
            channels=None,         # auto: [32, 64, 128, 256, 320]
            use_residuals=True,
        ),

        # ---- BYOL heads ----
        projector=dict(
            hidden_dim=2048,
            output_dim=256,
        ),
        predictor=dict(
            hidden_dim=2048,
        ),

        # ---- EMA schedule ----
        base_target_ema=_EMA_PRESETS[epoch_key],

        # ---- Optimizer: AdamW (not LARS) ----
        optimizer=dict(
            name='adamw',
            lr=_LR_PRESETS[epoch_key],
            weight_decay=_WD_PRESETS[epoch_key],
            betas=(0.9, 0.999),
        ),

        # ---- LR schedule ----
        lr_schedule=dict(
            warmup_epochs=10,
            warmup_steps=10 * steps_per_epoch,
            schedule='cosine',  # cosine decay after warmup
        ),

        # ---- Checkpointing ----
        checkpoint=dict(
            save_dir='/tmp/byol3d_checkpoints',
            save_every_epochs=50,
            filename='byol3d_pretrain.pt',
        ),

        # ---- Logging ----
        logging=dict(
            log_every_steps=10,
            eval_every_epochs=25,
        ),
    )

    return config


def get_smoke_test_config():
    """Tiny config for testing the pipeline without real data or GPU."""
    return dict(
        random_seed=42,
        num_epochs=2,
        batch_size=2,
        max_steps=4,
        steps_per_epoch=2,

        data=dict(
            data_dir=None,  # will use synthetic
            crop_size=8,
            volume_size=16,
            min_overlap=0.4,
            num_workers=0,
            pin_memory=False,
            in_channels=1,
        ),

        encoder=dict(
            in_channels=1,
            base_channels=8,
            num_levels=3,
            channels=None,
            use_residuals=True,
        ),

        projector=dict(hidden_dim=32, output_dim=16),
        predictor=dict(hidden_dim=32),

        base_target_ema=0.99,

        optimizer=dict(
            name='adamw',
            lr=1e-3,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
        ),

        lr_schedule=dict(
            warmup_epochs=0,
            warmup_steps=0,
            schedule='cosine',
        ),

        checkpoint=dict(
            save_dir='/tmp/byol3d_smoke',
            save_every_epochs=1,
            filename='smoke.pt',
        ),

        logging=dict(log_every_steps=1, eval_every_epochs=1),
    )