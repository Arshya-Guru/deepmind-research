#!/bin/bash
#SBATCH --job-name=byol-seg-ft
#SBATCH --output=seg_ft_%j.out
#SBATCH --error=seg_ft_%j.err
#SBATCH --time=1-00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --ntasks=1

# --- Why these resources: ---
# 1 GPU:  34 volumes × 8 crops = 272 samples/epoch at batch 2 = 136 steps.
#         Each step is one 128³ forward+backward. This is ~2min/epoch.
#         Multi-GPU would just starve for data.
# 64G RAM: 34 × 300³ × 4 bytes = ~3.6GB raw data, plus foreground
#          coordinate caches. 64G gives plenty of headroom for page cache.
# 8 CPUs:  Matching num_workers. The data is small enough that I/O
#          is never the bottleneck here.
# 4 hours: 200 epochs × ~2min = ~7 hours worst case, but bf16 + page cache
#          warming should keep it under 4h. Adjust if needed.

cd /nfs/khan/trainees/apooladi/abeta/deepmind-research

# Activate environment
# ---- Full fine-tuning with BYOL encoder ----
pixi run python -m byol3d.finetune_lightning \
    --checkpoint ./byol_work/8/byol3d_pretrain.pt \
    --data-dir ./nnssl/nnunet_data/preprocessed/Dataset001_ABetaPlaques \
    --save-dir ./seg_work/byol_ft \
    --fold 0 \
    --epochs 200 \
    --batch-size 2 \
    --lr 5e-4 \
    --encoder-lr-factor 0.1 \
    --crop-size 128 \
    --crops-per-volume 8 \
    --num-workers 8 \
    --precision bf16-mixed \
    --val-every 10

echo "Done: $(date)"
