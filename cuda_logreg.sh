#!/bin/bash
#SBATCH --job-name=cuda_logreg
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1               
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=cuda_logreg_%j.out

# Load conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml605

# Run the hardware-specific CUDA logistic regression
python cuda_main_logreg.py