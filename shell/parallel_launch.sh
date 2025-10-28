#!/bin/bash

# Launcher: run script.py directly with GNU Parallel

# Define number of parallel jobs
# max_jobs=$(( $(nproc) / 2 ))
max_jobs=16  

# Fixed parameters
alpha=50.0
teacher='He3'
loss='corr'
N_walkers=20
model='perceptron'
mode='online'


# Variable parameters
d_values=(250 500 1000 2000)
lr_values=(0.01 0.05 0.1)
students=('He3' 'relu')
variations=('None' 'twice')


echo '========================================'
echo 'Job started at:' $(date)
echo '========================================'
echo "Starting parallel jobs with max $max_jobs concurrent processes..."
echo " FIXED PARAMETERS:"
echo "  alpha = $alpha , teacher = $teacher , loss = $loss , N_walkers = $N_walkers , model = $model , mode = $mode"
echo " VARIABLE PARAMETERS:"
echo "  d values: ${d_values[@]}"
echo "  lr values: ${lr_values[@]}"
echo "  student types: ${students[@]}"
echo "  variation modes: ${variations[@]}"
echo ""



# Parallel execution
parallel -j $max_jobs \
  "python -u ./scripts/spikes_model.py --alpha ${alpha} --teacher ${teacher} --loss ${loss}\
  --N_walkers ${N_walkers} --model ${model} --mode ${mode} \
  --d {1} --lr {2} --student {3} --variation {4}> ./logs/run_repetita_iuvant_d{1}_lr{2}_student{3}_variation{4}.log 2>&1" \
  ::: "${d_values[@]}" ::: "${lr_values[@]}" ::: "${students[@]} ::: "${variations[@]}"
  
  
  
echo '========================================'
echo 'Job ended at:' $(date)
echo '========================================'