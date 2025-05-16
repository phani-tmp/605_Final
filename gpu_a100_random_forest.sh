#!/bin/bash
#SBATCH --job-name=model_gpu_a100
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=gpu_a100_rforest_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml605

python main.py --device cuda --model random_forest
