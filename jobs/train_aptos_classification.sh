#!/bin/bash
#SBATCH --job-name=train_aptos_classification
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=stampede
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# -----------------------------
# Activate Miniconda environment
# -----------------------------
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl_env

# --- Run training script ---
python train_resnet50.py \
    --dataset aptos2019 \
    --batch_size 32 \
    --epochs 80 \
    --lr 1e-4 \
    --input_size 224 \
    --num_workers 8 \
    --val_split 0.1