# import cvxpy as cp
# import numpy as np
# import pandas as pd
# import scipy.sparse as spa
from PEPit import PEP
from PEPit.functions import ConvexFunction, SmoothStronglyConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def test_PEPit_val(L, mu, rho, sigma, r, N=1):
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
    y0 = problem.set_initial_point()
    z0 = problem.set_initial_point()
    s0 = z0 + 1 / rho * y0

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)
    # to enforce feasibility of y0/z0 without initial constraints:
    _ = func(y0)
    _ = func(z0)

    x = x0
    y = y0
    z = z0
    s = s0
    x_vals = [x0]
    s_vals = [s0]
    for _ in range(N):
        x, _, _ = proximal_step(z - y, f1, sigma)
        y = y + x - z
        z, _, _ = proximal_step(x + y, f2, rho)
        s = z + 1 / rho * y
        x_vals.append(x)
        s_vals.append(s)

    problem.set_performance_metric((x_vals[-1] - x_vals[-2]) ** 2)

    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    print(pepit_tau)


def main():
    test_PEPit_val(10, 1, 1, 1, 1)


if __name__ == '__main__':
    main()
