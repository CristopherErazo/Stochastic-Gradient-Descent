#!/bin/bash
#SBATCH --partition=regular1,regular2

script_path=/home/cerazova/SGD/shell/
cd $script_path

for d in 100
do 
for alpha in 40
do
for lr in 0.005 0.01 0.05 0.1 0.5
do

echo "Running jobscript for d=$d alpha=$alpha lr=$lr"
sbatch jobscript.sh $d $alpha $lr
sleep 0.01

done
done
done 
