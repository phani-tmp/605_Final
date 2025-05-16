#!/bin/bash
#SBATCH --job-name=cpu_knn
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=cpu_knn_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml605

python main_cpu_opt_knn.py --threads $SLURM_CPUS_PER_TASK
