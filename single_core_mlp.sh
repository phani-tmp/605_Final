#!/bin/bash
#SBATCH --job-name=model_cpu1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=cpu1_mlp_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml605

python main.py --device cpu --model mlp
