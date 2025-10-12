#!/bin/bash
#SBATCH --job-name=isic2018_seg_det
#SBATCH --output=logs/isic2018_seg_det_%j.out
#SBATCH --error=logs/isic2018_seg_det_%j.err
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opencv_env

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="./segmentation/results/isic2018/deterministic/run_${TIMESTAMP}"
TRAIN_DIR="$RUN_DIR/train"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$TRAIN_DIR" "$EVAL_DIR" logs

python -m segmentation.scripts.train \
    --dataset isic2018 \
    --epochs 100 \
    --batch_size 16 \
    --num_workers 8 \
    --lr 1e-4 \
    --warmup_epochs 10 \
    --dropout 0.5 \
    --early_stop_patience 20 \
    --output_dir "$TRAIN_DIR" \
    --timestamp "$TIMESTAMP"

MODEL_PATH="$TRAIN_DIR/best_model.pth"

python -m segmentation.scripts.evaluate \
    --dataset isic2018 \
    --method deterministic \
    --model_path "$MODEL_PATH" \
    --batch_size 16 \
    --num_workers 8 \
    --dropout 0.5 \
    --output_dir "$EVAL_DIR"
