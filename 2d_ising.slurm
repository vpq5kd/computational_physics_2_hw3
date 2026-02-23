#!/bin/bash

#Submit this script with: sbatch myjob.slurm

#SBATCH --time=1:00:00   # job time limit
#SBATCH --nodes=1   # number of nodes
#SBATCH --ntasks-per-node=1   # number of tasks per node
#SBATCH --cpus-per-task=4   # number of CPU cores per task
#SBATCH --partition=standard   # partition
#SBATCH --account=cmsexo   # allocation name


python 2d_ising_analytics.py --melting_iterations 50 --measuring_iterations 100
