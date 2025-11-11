#!/bin/bash
#SBATCH --job-name=isic2018_edl
#SBATCH --output=logs/isic2018_edl_%j.out
#SBATCH --error=logs/isic2018_edl_%j.err
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --exclude=mscluster42,mscluster44,mscluster48,mscluster65

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opencv_env

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="./classification/results/isic2018/edl/run_${TIMESTAMP}"
TRAIN_DIR="$RUN_DIR/train"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$TRAIN_DIR" "$EVAL_DIR" logs

python -m classification.scripts.train \
    --dataset isic2018 \
    --num_classes 7 \
    --output_dir "$TRAIN_DIR" \
    --epochs 30 \
    --batch_size 32 \
    --num_workers 8 \
    --lr 1e-4 \
    --warmup_lr 1e-3 \
    --warmup_epochs 5 \
    --early_stop_patience 15 \
    --method edl \
    --zeta 0.1

MODEL_PATH="$TRAIN_DIR/model.pth"

python -m classification.scripts.evaluate \
    --dataset isic2018 \
    --num_classes 7 \
    --model_path "$MODEL_PATH" \
    --output_dir "$EVAL_DIR" \
    --batch_size 32 \
    --num_workers 8 \
    --method edl \
