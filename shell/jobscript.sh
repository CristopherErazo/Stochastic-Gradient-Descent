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

start_time=$(date +%s)
echo "Job started at $(date)"

# Define fixed parameters
snr=5.0 # Signal to noise ratio
alpha=70.0 # Time steps in units of d^(k-1)
teacher='He3' # Teacher model in perceptron
loss='corr' # Loss function
k=3 # Information exponent
echo "Fixed parameters: snr=$snr, alpha=$alpha, teacher=$teacher, loss=$loss, k=$k"

# Define input parameters
lr=$1
model=$2
echo "Input parameters: lr=$lr, model=$model"
echo "----------------------------------------"
echo "Current directory: $(pwd)"

# Define the parameter lists
d_values=(128 256 512 1024 2048)
modes=('online' 'repeat')
students=('He3' 'He2+He3')


# Loop over all combinations of parameters
for d in "${d_values[@]}"; do
    for mode in "${modes[@]}"; do
        for student in "${students[@]}"; do
            start_run=$(date +%s)
            echo "Starting run at $(date)"
            echo "Running with d=$d, mode=$mode, student=$student"
            python -u ./scripts/spikes_model.py --model $model --snr $snr --alpha $alpha \
            --teacher $teacher --loss $loss --k $k \
            --d $d --mode $mode --lr $lr --student $student > ./logs/run_"$model"_lr_"$lr"_d_"$d"_mode_"$mode"_student_"$student".log 2>&1
            end_run=$(date +%s)
            elapsed_run=$(( end_run - start_run ))
            echo "Run completed at $(date), took ${elapsed_run} seconds = $(( elapsed_run / 60 )) minutes"
            echo "----------------------------------------"
        done
    done
done


end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time so far: ${elapsed} seconds = $(( elapsed / 60 )) minutes"
