#!/bin/bash
#SBATCH --job-name=test_QC
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:02:00
#SBATCH -o /home/vranjan/algorithm-certification/experiments/control/quadcopter_data/test_QC%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=vranjan@princeton.edu

export PYTHONPATH='/home/vranjan/algorithm-certification/'

module purge
module load gurobi/9.5.2
module load anaconda3/2021.11
conda activate alg-cert

python3 test_mult_samples.py
