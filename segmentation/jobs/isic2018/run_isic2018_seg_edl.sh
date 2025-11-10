#!/bin/bash
#SBATCH --job-name=isic2018_seg_edl
#SBATCH --output=logs/isic2018_seg_edl_%j.out
#SBATCH --error=logs/isic2018_seg_edl_%j.err
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opencv_env

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="./segmentation/results/isic2018/edl/run_${TIMESTAMP}"
TRAIN_DIR="$RUN_DIR/train"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$TRAIN_DIR" "$EVAL_DIR" logs

python -m segmentation.scripts.train \
    --dataset isic2018 \
    --num_classes 2 \
    --output_dir "$TRAIN_DIR" \
    --epochs 10 \
    --batch_size 32 \
    --num_workers 8 \
    --lr 1e-4 \
    --early_stop_patience 20 \
    --method edl \
    --annealing_epochs 10 \
    --lambda_reg 0.01

MODEL_PATH="$TRAIN_DIR/best_model.pth"

python -m segmentation.scripts.evaluate \
    --dataset isic2018 \
    --num_classes 2 \
    --model_path "$MODEL_PATH" \
    --output_dir "$EVAL_DIR" \
    --batch_size 32 \
    --num_workers 8 \
    --method edl
