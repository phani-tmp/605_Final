#!/bin/bash
#SBATCH --job-name=logreg_cpu8
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=cpu8_%j.out

# Activate Conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate logreg-env

# Set thread environment variables
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8

# Run the training
python main.py --device cpu
