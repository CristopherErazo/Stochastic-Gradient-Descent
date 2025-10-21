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


# echo "first step"
# source ../.venv/bin/activate

cd ./shell/
start_time=$(date +%s)
# Define fixed parameters

d=$1
alpha=$2 # Time steps in units of d^(k-1)
k=4 # Information exponent of the problem
lr=$3


# Define the parameter lists
lr_values=($lr)

# Run single perceptron
echo "Running single perceptron purely online"

# Loop over all combinations of parameters
for lr in "${lr_values[@]}"; do
    echo "Running: alpha=$alpha, d=$d, p=$p, lr=$lr"
    python -u ../scripts/run_spike.py --spike 'False' --alpha $alpha --d $d  --lr $lr --k $k --student relu --dataset_size 0.0 --p_repeat 0.0 --mode online
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time so far: ${elapsed} seconds = $(( elapsed / 60 )) minutes"


echo "Running single perceptron with d^2 samples"

# Loop over all combinations of parameters
for lr in "${lr_values[@]}"; do
    echo "Running: alpha=$alpha, d=$d, p=$p, lr=$lr"
    python -u ../scripts/run_spike.py --spike 'False' --alpha $alpha --d $d  --lr $lr --k $k --student relu --dataset_size 1.0 --p_repeat 1.0 --mode repeat
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time so far: ${elapsed} seconds = $(( elapsed / 60 )) minutes"


# Run single perceptron
echo "Running spike purely online"

# Loop over all combinations of parameters
for lr in "${lr_values[@]}"; do
    echo "Running: alpha=$alpha, d=$d, p=$p, lr=$lr"
    python -u ../scripts/run_spike.py --spike 'True' --alpha $alpha --d $d  --lr $lr --k $k --student tanh --dataset_size 0.0 --p_repeat 0.0 --mode online
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time so far: ${elapsed} seconds = $(( elapsed / 60 )) minutes"


echo "Running spike with d^2 samples"

# Loop over all combinations of parameters
for lr in "${lr_values[@]}"; do
    echo "Running: alpha=$alpha, d=$d, p=$p, lr=$lr"
    python -u ../scripts/run_spike.py --spike 'True' --alpha $alpha --d $d  --lr $lr --k $k --student tanh --dataset_size 1.0 --p_repeat 1.0 --mode repeat
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time so far: ${elapsed} seconds = $(( elapsed / 60 )) minutes"
