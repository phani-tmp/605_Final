#!/bin/bash
#SBATCH --job-name=logreg_opt_cpu8
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=cpu_opt_logreg%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml605

# Threads are set inside the script via --threads
python main_cpu_opt_logreg.py --threads $SLURM_CPUS_PER_TASK
