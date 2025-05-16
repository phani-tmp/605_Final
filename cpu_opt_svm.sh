#!/bin/bash
#SBATCH --job-name=cpu_svm
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=cpu_svm_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml605

python main_cpu_opt_svm.py --threads $SLURM_CPUS_PER_TASK
