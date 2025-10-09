#!/bin/bash
#SBATCH --job-name=isic2018_run
#SBATCH --output=logs/isic2018_%j.out
#SBATCH --error=logs/isic2018_%j.err
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
RUN_DIR="./classification/results/isic2018/run_${TIMESTAMP}"
TRAIN_DIR="$RUN_DIR/train"
EVAL_DIR="$RUN_DIR/eval"
INFER_DIR="$RUN_DIR/inference"
mkdir -p "$TRAIN_DIR" "$EVAL_DIR" "$INFER_DIR" logs

# --- Train ---
echo "Starting training..."
python -m classification.scripts.train \
    --dataset isic2018 \
    --epochs 30 \
    --batch_size 32 \
    --num_workers 8 \
    --lr 1e-4 \
    --warmup_lr 1e-3 \
    --warmup_epochs 5 \
    --dropout 0.5 \
    --early_stop_patience 7 \
    --output_dir "$TRAIN_DIR"
    
# --- Model path ---
MODEL_PATH="$TRAIN_DIR/model.pth"

# --- Evaluate ---
echo "Starting evaluation..."
python -m classification.scripts.evaluate \
    --dataset isic2018 \
    --model_path "$MODEL_PATH" \
    --batch_size 32 \
    --num_workers 8 \
    --dropout 0.5 \
    --output_dir "$EVAL_DIR"

# --- Uncertainty Inference ---
echo "Starting uncertainty inference..."
python -m classification.scripts.ue_inference \
    --dataset isic2018 \
    --model_path "$MODEL_PATH" \
    --batch_size 32 \
    --num_workers 8 \
    --dropout 0.5 \
    --mc_samples 20 \
    --output_dir "$INFER_DIR"

echo "✅ Run completed successfully."
echo "📂 All results saved in: $RUN_DIR"
