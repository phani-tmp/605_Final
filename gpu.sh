#!/bin/bash
#SBATCH --job-name=logreg_gpu
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=gpu_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate logreg-env

python main.py --device cuda
