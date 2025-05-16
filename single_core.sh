#!/bin/bash
#SBATCH --job-name=logreg_cpu1             # Name of your job
#SBATCH --cpus-per-task=1                  # Request 1 CPU core
#SBATCH --mem=4G                           # Memory allocation
#SBATCH --time=00:15:00                    # Max job time
#SBATCH --output=cpu1_%j.out               # Output log file

# Activate your conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate logreg-env

# Run the Python script using CPU
python main.py --device cpu
