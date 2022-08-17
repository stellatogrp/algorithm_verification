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
    r_vals = [1.015, 2.367, 4.436, 5.694, 6.601, 7.458, 11.934, 17.999, 20.969, 24.352]
    for r in r_vals:
        res = test_PEPit_val(L, mu, t, r, N=N)
        print(r, res)
    # print(res)

    # 5.0, 10.0, 1.0154222689416668, 0.6841392815228748
    # 10.0, 20.0, 2.367989324851323, 3.7233392552346736
    # 25.0, 50.0, 4.4362865301235885, 13.064193102477471
    # 50.0, 100.0, 5.694442110588323, 21.527265520584457
    # 75.0, 150.0, 6.600936194119201, 28.9277654583038
    # 100.0, 200.0, 7.457511545179829, 36.93392583310136
    # 250.0, 500.0, 11.933780862103053, 94.538571937282
    # 500.0, 1000.0, 17.998979915359264, 146.90897572063884
    # 750.0, 1500.0, 20.96925021756563, 341.7809602093723
    # 1000.0, 2000.0, 24.35158189730156, 545.0529989864015


if __name__ == '__main__':
    main()
