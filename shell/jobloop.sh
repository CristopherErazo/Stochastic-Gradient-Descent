#!/bin/bash
# SBATCH --partition=regular1,regular2

# script_path=/home/cerazova/SGD/shell/
# cd $script_path

echo "Starting jobloop.sh"

lr_values=(0.05 0.1 0.5 1.0)
models=('perceptron' 'skewed')

for model in "${models[@]}"; do
    for lr in "${lr_values[@]}"; do
        echo "Submitting jobscript for model=$model lr=$lr"
        bash ./shell/jobscript.sh $lr $model > ./logs/job_"$model"_lr"$lr".log 2>&1 &
        sleep 0.01
    done
done
