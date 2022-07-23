import numpy as np
import cvxpy as cp
import os
import joblib
import pandas as pd

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction, ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step

from scipy.stats import ortho_group


def test_PEPit_val(L, mu, t, r, N=1):
    problem = PEP()
    verbose = 0

    # Declare a strongly convex smooth function and a closed convex proper function
    f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    f2 = problem.declare_function(ConvexFunction)
    func = f1 + f2

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)

    # Run the proximal gradient method starting from x0
    x = x0
    x_vals = [x0]
    for _ in range(N):
        y = x - t * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, t)
        x_vals.append(x)

    # Set the performance metric to the distance between x and xs
    problem.set_performance_metric((x_vals[-1] - x_vals[-2]) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau


def main():
    np.random.seed(0)

    mu = 1
    L = 10
    t = 2 / (mu + L)
    N = 5
    r = 172.7800281379808
    res = test_PEPit_val(L, mu, t, r, N=N)
    print(res)


if __name__ == '__main__':
    main()
