#!/bin/bash
# Minimal launcher: run spikes_model.py directly with GNU Parallel

lr_values=(0.2)
models=('perceptron')
modes=('online' 'repeat')
d_values=(10 20 30)   # example, if your jobscript loops over d_values


max_jobs=$(nproc)   # use all logical CPUs
# or safer: 80% of physical cores
max_jobs=$(( $(nproc) / 2 ))

# max_jobs=48  # number of concurrent jobs

# Parallel execution
parallel -j $max_jobs \
  "python -u ./scripts/spikes_model.py --datasize 100 --model {2} --snr 5.0 --alpha 70.0 --teacher He --loss mse --k 1 --d {4} --mode {3} --lr {1} --student studentA > ./logs/run_{2}_lr{1}_mode{3}_d{4}.log 2>&1" \
  ::: "${lr_values[@]}" ::: "${models[@]}" ::: "${modes[@]}" ::: "${d_values[@]}"
