#!/bin/bash
#SBATCH --job-name=aptos_edl
#SBATCH --output=logs/aptos_edl_%j.out
#SBATCH --error=logs/aptos_edl_%j.err
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# --- Load conda environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate opencv_env

# --- Run directories ---
TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="./classification/results/aptos2019/edl_run_${TIMESTAMP}"
TRAIN_DIR="$RUN_DIR/train"
EVAL_DIR="$RUN_DIR/eval"
mkdir -p "$TRAIN_DIR" "$EVAL_DIR" logs

# --- Train ---
echo "🚀 Starting EDL training..."
python -m classification.scripts.train \
    --dataset aptos2019 \
    --method edl \
    --epochs 30 \
    --batch_size 32 \
    --num_workers 8 \
    --lr 1e-4 \
    --warmup_lr 1e-3 \
    --warmup_epochs 5 \
    --early_stop_patience 7 \
    --output_dir "$TRAIN_DIR"

# --- Model path ---
MODEL_PATH="$TRAIN_DIR/model.pth"

# --- Evaluate ---
echo "📊 Evaluating EDL model..."
python -m classification.scripts.evaluate \
    --dataset aptos2019 \
    --method edl \
    --model_path "$MODEL_PATH" \
    --batch_size 32 \
    --num_workers 8 \
    --output_dir "$EVAL_DIR"

echo "⚡ Skipping ue_inference — EDL provides intrinsic uncertainty."
echo "✅ EDL run completed successfully."
echo "📂 Results saved in: $RUN_DIR"
