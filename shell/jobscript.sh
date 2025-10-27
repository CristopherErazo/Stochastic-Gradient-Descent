#!/bin/bash

# SBATCH --job-name=sgd
# SBATCH --nodes=1
# SBATCH --ntasks=40
# SBATCH --time=12:00:00
# SBATCH --mem=20G
# SBATCH --partition=regular1,regular2
# SBATCH --qos=fastlane # for debugging
# SBATCH --output=../logs/job-%j.out
# SBATCH --error=../logs/job-%j.err


# Define fixed parameters
snr=5.0 # Signal to noise ratio
alpha=2.0 # Time steps in units of d^(k-1)
teacher='He3' # Teacher model in perceptron
loss='corr' # Loss function
k=3 # Information exponent
datasize=5.0 # Dataset size in units of d
student='He3' # Student model
echo "Fixed parameters: snr=$snr, alpha=$alpha, teacher=$teacher, loss=$loss, k=$k ,datasize=$datasize"

start_time=$(date +%s)
echo "Job started at $(date)"

# Define input parameters
lr=$1
model=$2
mode=$3

echo "Input parameters: lr=$lr, model=$model"
echo "----------------------------------------"
echo "Current directory: $(pwd)"

# Define the parameter lists
d_values=(100 200 400 800)



# Loop over all combinations of parameters
for d in "${d_values[@]}"; do
    start_run=$(date +%s)
    echo "------------    RUNNING with d=$d    ------------"
    echo "Starting run at $(date)"
    python -u ./scripts/spikes_model.py --datasize $datasize --model $model --snr $snr --alpha $alpha \
    --teacher $teacher --loss $loss --k $k \
    --d $d --mode $mode --lr $lr --student $student > ./logs/run_"$model"_lr_"$lr".log 2>&1
    end_run=$(date +%s)
    elapsed_run=$(( end_run - start_run ))
    echo "Run completed at $(date), took ${elapsed_run} seconds = $(( elapsed_run / 60 )) minutes"
    echo "----------------------------------------"
    echo ""
done


end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time so far: ${elapsed} seconds = $(( elapsed / 60 )) minutes"
