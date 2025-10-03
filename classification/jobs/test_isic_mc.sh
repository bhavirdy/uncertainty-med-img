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

# --- Inference with UE ---
python -m classification.scripts.ue_inference \
    --config "./classification/configs/test_infer_config.yaml"

echo "Run completed."
