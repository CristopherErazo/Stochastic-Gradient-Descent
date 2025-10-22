#!/bin/bash

#Input Parameters
d=$1
alpha=$2
k=4
loss='corr'

cd ./shell/

# Maximum number of concurrent jobs
MAX_JOBS=8
job_count=0

# Parameter list

lr_values=(0.01 0.05 0.1 0.5)

# Parameter combinations: student, dataset_size, p_repeat, mode
combinations=(
    "relu 0.0 0.0 online"
    "relu 1.0 1.0 repeat"
    "tanh 0.0 0.0 online"
    "tanh 1.0 1.0 repeat"
)

for lr in "${lr_values[@]}"; do
    for combo in "${combinations[@]}"; do
        echo "Running for lr = $lr and combo = $combo"
        start_time=$(date +%s)
        # Split the combo string into variables
        read student dataset_size p_repeat mode <<< "$combo"

        # Run the Python script in background
        python -u ../scripts/run_spike.py --alpha $alpha --d $d  --lr $lr --k $k \
        --spike "True" --student $student --dataset_size $dataset_size \
        --p_repeat $p_repeat --mode $mode > ../logs/combination_st_"$student"_mod_"$mode"_lr_"$mode".log 2>&1 &
        
        end_time=$(date +%s)
        elapsd=$(( end_time - start_time ))
        echo "Total time so far: ${elapsed} seconds = $(( elapsed / 60 )) minutes"


        ((job_count++))

        # Limit number of concurrent jobs
        if (( job_count >= MAX_JOBS )); then
            wait
            job_count=0
        fi

    done
done
# Wait for any remaining jobs
wait


