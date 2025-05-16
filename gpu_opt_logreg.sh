#!/bin/bash
#SBATCH --job-name=logreg_opt_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1            # 🔥 Specify GPU type here (a100 or h100)
#SBATCH --cpus-per-task=6
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=gpu_opt_logreg_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml605

python main_gpu_opt_logreg.py
