#!/bin/bash

# ==============================================
# SAFE JOB SUBMISSION SCRIPT
# Only submits individual job scripts
# ==============================================

# List of SPECIFIC scripts we want to submit
SCRIPTS_TO_RUN=(
    # GPU A100 scripts
    "gpu_a100_decision_tree.sh"
    "gpu_a100_extra_trees.sh"
    "gpu_a100_knn.sh"
    "gpu_a100_logreg.sh"
    "gpu_a100_mlp.sh"
    "gpu_a100_naive_bayes.sh"
    "gpu_a100_random_forest.sh"
    "gpu_a100_svm.sh"
    
    # GPU H100 scripts
    "gpu_h100_decision_tree.sh"
    "gpu_h100_extra_trees.sh"
    "gpu_h100_knn.sh"
    "gpu_h100_logreg.sh"
    "gpu_h100_mlp.sh"
    "gpu_h100_naive_bayes.sh"
    "gpu_h100_random_forest.sh"
    "gpu_h100_svm.sh"
    
    # Multi-core scripts
    "multi_core_decision_tree.sh"
    "multi_core_extra_trees.sh"
    "multi_core_knn.sh"
    "multi_core_logreg.sh"
    "multi_core_mlp.sh"
    "multi_core_naive_bayes.sh"
    "multi_core_random_forest.sh"
    "multi_core_svm.sh"
    
    # Single-core scripts
    "single_core_decision_tree.sh"
    "single_core_extra_trees.sh"
    "single_core_knn.sh"
    "single_core_logreg.sh"
    "single_core_mlp.sh"
    "single_core_naive_bayes.sh"
    "single_core_random_forest.sh"
    "single_core_svm.sh"
)

# Submit each script explicitly
for script in "${SCRIPTS_TO_RUN[@]}"; do
    if [[ -f "$script" ]]; then
        echo "Submitting $script"
        sbatch "$script"
        sleep 0.1
    else
        echo "WARNING: Script $script not found!"
    fi
done

echo "All jobs submitted successfully"
