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
RUN_DIR="./classification/results/aptos2019/run_${TIMESTAMP}"
mkdir -p "$RUN_DIR/train" "$RUN_DIR/eval" "$RUN_DIR/inference"

# --- Prepare run-specific configs ---
TRAIN_CONFIG="$RUN_DIR/train_config.yaml"
EVAL_CONFIG="$RUN_DIR/eval_config.yaml"
# INFER_CONFIG="$RUN_DIR/infer_config.yaml"

sed "s|{OUTPUT_DIR}|$RUN_DIR/train|g" ./classification/configs/aptos_train_config.yaml > "$TRAIN_CONFIG"

# --- Train ---
python -m classification.scripts.train \
    --config "$TRAIN_CONFIG"

# --- Create eval config with model path ---
MODEL_PATH=$RUN_DIR/train/model.pth
sed "s|{OUTPUT_DIR}|$RUN_DIR/eval|g; s|{MODEL_PATH}|$MODEL_PATH|g" \
    ./classification/configs/aptos_eval_config.yaml > "$EVAL_CONFIG"

# --- Evaluate ---
python -m classification.scripts.evaluate \
    --config "$EVAL_CONFIG"

# # --- Create inference config with model path ---
# sed "s|{OUTPUT_DIR}|$RUN_DIR/inference|g; s|{MODEL_PATH}|$MODEL_PATH|g" \
#     ./classification/configs/aptos_infer_config.yaml > "$INFER_CONFIG"

# --- Inference with UE ---
# python -m classification.scripts.ue_inference \
#     --config "$INFER_CONFIG" \
#     --model_path "$MODEL_PATH" \

echo "Run completed. All results saved in $RUN_DIR"
