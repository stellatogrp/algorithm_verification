# Algorithm Verification
This repository is by [Vinit Ranjan](https://vinitranjan1.github.io/) and [Bartolomeo Stellato](https://stellato.io/) and contains the Python source code to reproduce experiments in our paper [Verification of First-Order Methods for Parametric Quadratic Optimization](https://arxiv.org/pdf/2403.03331.pdf).

# Abstract
We introduce a numerical framework to verify the finite step convergence of first-order methods for parametric convex quadratic optimization. We formulate the verification problem as a mathematical optimization problem where we maximize a performance metric (e.g., fixed-point residual at the last iteration) subject to constraints representing proximal algorithm steps (e.g., linear system solutions, projections, or gradient steps). Our framework is highly modular because we encode a wide range of proximal algorithms as variations of two primitive steps: affine steps and element-wise maximum steps. Compared to standard convergence analysis and performance estimation techniques, we can explicitly quantify the effects of warm-starting by directly representing the sets where the initial iterates and parameters live. We show that the verification problem is NP-hard, and we construct strong semidefinite programming relaxations using various constraint tightening techniques. Numerical examples in nonnegative least squares, network utility maximization, Lasso, and optimal control show a significant reduction in pessimism of our framework compared to standard worst-case convergence analysis techniques.

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
silver_nonstrong_NNLS
silver_strongcvx_NNLS
```

### Results
For each experiment, the results are saved in the `data/` subfolder.
The results include the SDP objective value, solve/setup times, and some other auxiliary information about the size of the experiment and other algorithm parameters.
Depending on the SDP size, setting up and solving the problems can take some time but the `tqdm` package is included to allow easy progress tracking for the purely Python parts.
