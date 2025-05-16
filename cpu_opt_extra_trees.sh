#!/bin/bash
#SBATCH --job-name=cpu_extra_trees
#SBATCH --cpus-per-task=8
#SBATCH --mem=8G
#SBATCH --time=00:10:00
#SBATCH --output=cpu_extra_trees_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml605

python main_cpu_opt_extra_trees.py --threads $SLURM_CPUS_PER_TASK
