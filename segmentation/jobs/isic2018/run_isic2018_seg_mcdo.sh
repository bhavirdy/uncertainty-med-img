#!/bin/bash
#SBATCH --job-name=isic2018_seg_mcdo
#SBATCH --output=logs/isic2018_seg_mcdo_%j.out
#SBATCH --error=logs/isic2018_seg_mcdo_%j.err
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opencv_env

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="./segmentation/results/isic2018/mcdo/run_${TIMESTAMP}"
TRAIN_DIR="$RUN_DIR/train"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$TRAIN_DIR" "$EVAL_DIR" logs

python -m segmentation.scripts.train \
    --dataset isic2018 \
    --num_classes 2 \
    --epochs 100 \
    --batch_size 16 \
    --num_workers 8 \
    --lr 1e-4 \
    --dropout 0.5 \
    --early_stop_patience 20 \
    --output_dir "$TRAIN_DIR" \

MODEL_PATH="$TRAIN_DIR/best_model.pth"

python -m segmentation.scripts.evaluate \
    --dataset isic2018 \
    --num_classes 2 \
    --method mcdo \
    --model_path "$MODEL_PATH" \
    --batch_size 16 \
    --num_workers 8 \
    --dropout 0.5 \
    --mc_samples 20 \
    --output_dir "$EVAL_DIR"
