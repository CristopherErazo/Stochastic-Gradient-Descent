#!/bin/bash

#Input Parameters
d=$1
alpha=$2
lr=$3
k=4

# Maximum number of concurrent jobs
MAX_JOBS=8
job_count=0

# Parameter combinations: spike, student, dataset_size, p_repeat, mode
combinations=(
    "False relu 0.0 0.0 online"
    "False relu 1.0 1.0 repeat"
    "True tanh 0.0 0.0 online"
    "True tanh 1.0 1.0 repeat"
)

for combo in "${combinations[@]}"; do
    # Split the combo string into variables
    read spike student dataset_size p_repeat mode <<< "$combo"

    # Run the Python script in background
    python -u ../scripts/run_spike.py --alpha $alpha --d $d  --lr $lr --k $k \
    --spike "$spike" --student "$student" --dataset_size "$dataset_size" \
    --p_repeat "$p_repeat" --mode "$mode" > ../logs/combination.log 2>&1 &

    ((job_count++))

    # Limit number of concurrent jobs
    if (( job_count >= MAX_JOBS )); then
        wait
        job_count=0
    fi

done

# Wait for any remaining jobs
wait


