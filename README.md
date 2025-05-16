Here's the complete, properly formatted `README.md` file in a clean notepad-style format that you can copy and paste directly:

```
# 605 Final Project – Benchmarking ML Models on Zaratan (CPU/GPU)

This project benchmarks various ML models on the CIFAR-100 dataset using:
- Single-core CPU
- Multi-core CPU (OneDNN/Thread-optimized)
- GPU (PyTorch + AMP)
- Custom CUDA kernel (Numba)

## Project Structure

| File Type               | Description                                     |
|-------------------------|-------------------------------------------------|
| main.py                | Unified benchmark runner for all models         |
| main_cpu_opt_*.py      | Optimized CPU scripts (OneDNN/Threads)          |
| main_gpu_opt_*.py      | Optimized GPU scripts (AMP, cuDNN)              |
| cuda_main_logreg.py    | CUDA kernel (Numba) for logistic regression     |
| *.sh files             | SLURM job scripts for Zaratan cluster           |
| submit_all.sh          | Batch script to run all single/multicore jobs   |
| requirements.txt       | Required dependencies                           |

## Setup

1. Clone the repo:
```bash
git clone https://github.com/phani-tmp/605_Final.git
cd 605_Final
```

2. Install dependencies (only for local testing):
```bash
pip install -r requirements.txt
```

## Download CIFAR-100 Dataset

Before running any training jobs, download CIFAR-100 once:
```bash
python download_cifar.py
```

Dataset will be stored in:
```
./cifar100_data/
```

## Run All Benchmarks

Includes 8 models:
- logreg
- mlp
- svm
- knn
- naive_bayes
- decision_tree
- random_forest
- extra_trees

### How to Run:
```bash
chmod +x *.sh
./submit_all.sh
```

Monitor jobs:
```bash
squeue -u $USER
```

## CPU-Optimized Versions

Run individual models with tuned CPU threads:
```bash
sbatch cpu_opt_logreg.sh
sbatch cpu_opt_mlp.sh
sbatch cpu_opt_svm.sh
# ... and so on for other models
```

## GPU-Optimized Versions

Available for logreg and mlp with AMP/cuDNN:
```bash
sbatch gpu_opt_logreg.sh
sbatch gpu_opt_mlp.sh
```

## CUDA Kernel Benchmark

Special Numba implementation of logistic regression:
```bash
sbatch cuda_logreg.sh
```

Features:
- Custom CUDA matrix operations
- GPU utilization metrics
- Memory usage reporting

## Checking Results

View output logs after job completion:
```bash
ls *.out
cat cpu_opt_logreg_<jobid>.out
cat gpu_opt_logreg_<jobid>.out
cat cuda_logreg_<jobid>.out
```

## Manual Execution

Run models directly:
```bash
python main.py --device cpu --model svm
python main.py --device cuda --model mlp
```

## Notes

- Dataset excluded via .gitignore
- Output files (*.out) should not be committed
- Extend by adding models to main.py/get_model()
```

