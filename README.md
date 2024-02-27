# Algorithm Verification
This repository is by [Vinit Ranjan](https://vinitranjan1.github.io/) and [Bartolomeo Stellato](https://stellato.io/) and contains the Python source code to reproduce experiments in our paper "Verification of First-Order Methods for Parametric Quadratic Optimization."

If you find this repository helpful in your work, please consider citing our papers (FILL ARXIV LINK)

# Abstract
Fill in once finalized...

## Installation
To install the package, run
```
$ pip install git+https://github.com/stellatogrp/algorithm_verification
```

## Packages
The main required packages are
```
cvxpy >= 1.2.0
gurobipy
Mosek
tqdm
PEPit
```
Both Mosek and Gurobi require licenses, but free academic licenses for individual use can be obtained from their respective websites.

### Running experiments
Experiments for the paper should be run from the `paper_experiments/` folder with the command:
```
python <example>_experiment.py
```
where ```<example>``` is one of the following:
```
ISTA
MPC
NNLS
NUM
silver
```

### Results
For each experiment, the results are saved in the `data/` subfolder.
The results include the SDP objective value, solve/setup times, and some other auxiliary information about the size of the experiment and other algorithm parameters.
Depending on the SDP size, setting up and solving the problems can take some time but the `tqdm` package is included to allow easy progress tracking for the purely Python parts.
