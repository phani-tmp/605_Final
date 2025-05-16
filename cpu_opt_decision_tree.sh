#!/bin/bash
#SBATCH --job-name=cpu_decision_tree
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=cpu_decision_tree_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml605

python main_cpu_opt_decision_tree.py --threads $SLURM_CPUS_PER_TASK
