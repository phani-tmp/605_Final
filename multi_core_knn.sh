#!/bin/bash
#SBATCH --job-name=model_cpu8
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=cpu8_knn_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml605

export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

python main.py --device cpu --model knn
