#!/bin/bash
#SBATCH --job-name=test_NNLS_gurobi
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00
#SBATCH -o /home/vranjan/experiments/NNLS/data/test_NNLS_multistepfix_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=vranjan@princeton.edu

export PYTHONPATH='/home/vranjan'

module purge
module load gurobi/9.0.1
module load anaconda3/2021.11
conda activate alg-cert

python3 test_NNLS_multstep.py
