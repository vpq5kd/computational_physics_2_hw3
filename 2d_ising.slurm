#!/bin/bash

#Submit this script with: sbatch myjob.slurm

#SBATCH --time=12:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=16   # number of CPU cores per task
#SBATCH --partition=standard   # partition
#SBATCH --constraint=rivanna   # cluster
#SBATCH --account=phys56xx   # allocation name


python 2d_ising.py
