#!/bin/bash
#SBATCH --job-name=isic2018_seg_demo
#SBATCH --output=logs/isic2018_seg_demo_%j.out
#SBATCH --error=logs/isic2018_seg_demo_%j.err
#SBATCH --partition=stampede
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate opencv_env

python -m segmentation.demo