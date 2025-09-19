#!/bin/bash
#SBATCH --job-name=aptos_run
#SBATCH --output=logs/aptos_%j.out
#SBATCH --error=logs/aptos_%j.err
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00

# --- Load conda environment ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate dl_env

TIMESTAMP=$(date +"%Y%m%d-%H%M%S")
RUN_DIR="./classification/results/${DATASET}/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR/train" "$RUN_DIR/eval" "$RUN_DIR/inference"

# --- Train ---
python /classification/scripts/train.py \
    --config ./classification/configs/aptos_train_config.yaml

MODEL_PATH=$RUN_DIR/train/model.pth

# --- Evaluate ---
python /classification/scripts/evaluate.py \
    --config ./classification/configs/aptos_eval_config.yaml \
    --model_path "$MODEL_PATH" \

# --- Inference with UE ---
# python classification/scripts/ue_inference.py \
#     --dataset $DATASET \
#     --model_path "$MODEL_PATH" \
#     --out_dir "$RUN_DIR/inference" \
#     --mcdo \
#     --forward_passes 30

echo "Run completed. All results saved in $RUN_DIR"
