# 605 Final Project – Benchmarking ML Models on Zaratan (CPU/GPU)

This project benchmarks various ML models on the CIFAR-100 dataset using:
- **Single-core CPU**
- **Multi-core CPU (OneDNN/Thread-optimized)**
- **GPU (PyTorch + AMP)**
- **Custom CUDA kernel (Numba)**

---

##  Project Structure

| File Type               | Description                                     |
|-------------------------|-------------------------------------------------|
| `main.py`              | Unified benchmark runner for all models         |
| `main_cpu_opt_*.py`    | Optimized CPU scripts (OneDNN/Threads)          |
| `main_gpu_opt_*.py`    | Optimized GPU scripts (AMP, cuDNN)              |
| `cuda_main_logreg.py`  | CUDA kernel (Numba) for logistic regression     |
| `.sh` files            | SLURM job scripts for Zaratan cluster           |
| `submit_all.sh`        | Batch script to run all single/multicore jobs   |
| `requirements.txt`     | Required dependencies                           |

---

##  Setup

1. **Clone the repo**
   ```bash
   git clone https://github.com/phani-tmp/605_Final.git
   cd 605_Final
Install dependencies (only for local testing)

```bash
 pip install -r requirements.txt
Download CIFAR-100 Dataset
Before running any training jobs, download CIFAR-100 once:

```bash
python download_cifar.py

Dataset is stored inside:

./cifar100_data/
 Run All Single-Core & Multi-Core Benchmarks
These include 8 models:

logreg, mlp, svm, knn, naive_bayes, decision_tree, random_forest, extra_trees

 How to Run:
```bash
chmod +x *.sh
./submit_all.sh

Monitor with:

```bash
squeue -u $USER
Run CPU-Optimized Versions
Run each model individually with tuned CPU threads:

```bash
sbatch cpu_opt_logreg.sh
sbatch cpu_opt_mlp.sh
sbatch cpu_opt_svm.sh
...

Run GPU-Optimized Versions (2 Models Only)
Only logreg and mlp are tuned with AMP and cuDNN:

```bash
sbatch gpu_opt_logreg.sh
sbatch gpu_opt_mlp.sh
 Run CUDA Kernel Benchmark (Hardware-Specific)
A special CUDA version of logistic regression is implemented using Numba kernels.

To run:

```bash
sbatch cuda_logreg.sh
This executes:

Pure CUDA matrix multiplication, sigmoid, gradient updates

Reports GPU utilization, training time, memory used

 View Output Logs
After jobs complete:

```bash
ls *.out
cat cpu_opt_logreg_<jobid>.out
cat gpu_opt_logreg_<jobid>.out
cat cuda_logreg_<jobid>.out
Custom Manual Runs
Run any model manually:

```bash
python main.py --device cpu --model svm
python main.py --device cuda --model mlp
 Notes
cifar100_data/ is excluded from GitHub using .gitignore

Easy to extend: Just add your model logic to main.py or get_model()
