#!/bin/bash

# Define fixed parameters
d=300
alpha=1200

# Define the parameter lists
# p_values=(0.0 2.0 4.0 8.0 16.0 32.0)
lr_values=(4.0)
k0_values=(1.0 1.5 2.0 2.5 3.0)

start_time=$(date +%s)
# Loop over all combinations of parameters
for k0 in "${k0_values[@]}"; do
    for lr in "${lr_values[@]}"; do
        echo "Running: alpha=$alpha, d=$d, p=$p, lr=$lr"
        python ./scripts/multi_walk.py --alpha $alpha --d $d --p_repeat 0.8 --k0 $k0 --lr $lr --student='tanh' --teacher='He3' --k=3 --loss='corr' 
    done
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "Total time: ${elapsed} seconds"

echo "Total time: $(( elapsed / 60 )) minutes"
