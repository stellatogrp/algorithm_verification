#!/bin/bash
#SBATCH --job-name=test_NNLS
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G
#SBATCH --time=02:30:00
#SBATCH -o /home/vranjan/algorithm-certification/paper_experiments/NNLS/data/test_NNLS_%A.txt
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=vranjan@princeton.edu

export PYTHONPATH='/home/vranjan/algorithm-certification'
module purge
module load gurobi/10.0.1
module load anaconda3/2023.9
# module load anaconda3/2023.9 cudnn/cuda-11.x/8.2.0 cudatoolkit/11.3 nvhpc/21.5
conda activate alg-cert

python NNLS_experiment.py