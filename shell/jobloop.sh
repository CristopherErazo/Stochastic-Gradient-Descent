#!/bin/bash
# SBATCH --partition=regular1,regular2

# script_path=/home/cerazova/SGD/shell/
# cd $script_path

echo "Starting jobloop.sh"

lr_values=(0.2)
models=('perceptron')
modes=('online' 'repeat')

for mode in "${modes[@]}"; do
    for model in "${models[@]}"; do
        for lr in "${lr_values[@]}"; do
            echo "Submitting jobscript for model=$model lr=$lr mode=$mode"
            bash ./shell/jobscript.sh $lr $model $mode > ./logs/job_"$model"_lr"$lr"_mode"$mode".log 2>&1 &
            sleep 0.01
        done
    done
done