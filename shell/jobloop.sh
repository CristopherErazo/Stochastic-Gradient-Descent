#!/bin/bash
#SBATCH --partition=regular1,regular2



for d in 250 500 1000
do 
for alpha in 5 10 20
do

echo "Running jobscript for d=$d alpha=$alpha"
sbatch jobscript.sh $d $alpha
sleep 0.01

done
done 
