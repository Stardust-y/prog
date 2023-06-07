#!/bin/bash
for env in 5 6
do
  for time in 1 2 3
  do
    python plan_generation.py $env $time
    python prog_script.py $env $time
    python preprocess.py $env $time
    python prog_exec.py --graph_num $env --exp_times $time
  done
done