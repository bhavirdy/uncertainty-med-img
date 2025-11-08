#!/bin/bash
#SBATCH --job-name=aptos2019_det
#SBATCH --output=logs/aptos2019_det_%j.out
#SBATCH --error=logs/aptos2019_det_%j.err
#SBATCH --partition=bigbatch
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opencv_env

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="./classification/results/aptos2019/deterministic/run_${TIMESTAMP}"
TRAIN_DIR="$RUN_DIR/train"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$TRAIN_DIR" "$EVAL_DIR" logs

# --- Train ---
python -m classification.scripts.train \
    --dataset aptos2019 \
    --num_classes 5 \
    --epochs 30 \
    --batch_size 32 \
    --num_workers 8 \
    --lr 1e-4 \
    --warmup_lr 1e-3 \
    --warmup_epochs 5 \
    --dropout 0.5 \
    --early_stop_patience 10 \
    --output_dir "$TRAIN_DIR" 

MODEL_PATH="$TRAIN_DIR/model.pth"

# --- Evaluate ---
python -m classification.scripts.evaluate \
    --dataset aptos2019 \
    --num_classes 5 \
    --method deterministic \
    --model_path "$MODEL_PATH" \
    --batch_size 32 \
    --num_workers 8 \
    --dropout 0.5 \
    --output_dir "$EVAL_DIR"
