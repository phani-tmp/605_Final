#!/bin/bash
#SBATCH --job-name=test_cuda
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=test_cuda_%j.out

source ~/miniconda3/etc/profile.d/conda.sh
conda activate ml605

python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('CUDA Device:', torch.cuda.get_device_name(0))"
