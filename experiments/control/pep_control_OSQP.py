# import cvxpy as cp
# import numpy as np
# import pandas as pd
# import scipy.sparse as spa
from PEPit import PEP
from PEPit.functions import ConvexFunction, SmoothStronglyConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step

# from PEPit.examples.composite_convex_minimization.douglas_rachford_splitting import *


# def test_PEPit_val(L, mu, rho, sigma, r, N=1):
#     problem = PEP()
#     verbose = 0

#     # Declare a convex and a smooth convex function.
#     func1 = problem.declare_function(ConvexFunction)
#     # func2 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
#     func2 = problem.declare_function(SmoothConvexFunction, L=L)
#     # Define the function to optimize as the sum of func1 and func2
#     func = func1 + func2

#     # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
#     xs = func.stationary_point()
#     fs = func(xs)

#     # Then define the starting point x0 of the algorithm and its function value f0
#     x0 = problem.set_initial_point()

#     # Compute n steps of the Douglas-Rachford splitting starting from x0
#     x = [x0 for _ in range(N + 1)]
#     w = [x0 for _ in range(N + 1)]
#     for i in range(N):
#         x[i + 1], _, _ = proximal_step(w[i], func2, rho)
#         y, _, fy = proximal_step(2 * x[i + 1] - w[i], func1, sigma)
#         w[i + 1] = w[i] + (y - x[i + 1])

#     # Set the initial constraint that is the distance between x0 and xs = x_*
#     problem.set_initial_condition((x[0] - xs) ** 2 <= r ** 2)

#     # Set the performance metric to the final distance to the optimum in function values
#     # problem.set_performance_metric((x[-1] - x[-2]) ** 2)
#     problem.set_performance_metric((func2(y) + fy) - fs)

#     # Solve the PEP
#     pepit_verbose = max(verbose, 0)
#     pepit_tau = problem.solve(verbose=pepit_verbose)


#     print(pepit_tau)

def test_admm_pep(L, mu, alpha, theta, N=1):
    verbose = 0
    # Instantiate PEP
    problem = PEP()

    # Declare a convex and a smooth convex function.
    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    # fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Compute n steps of the Douglas-Rachford splitting starting from x0
    x = [x0 for _ in range(N)]
    w = [x0 for _ in range(N + 1)]
    for i in range(N):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x[0] - xs) ** 2 <= 1)

    # Set the performance metric to the final distance to the optimum in function values
    # problem.set_performance_metric((func2(y) + fy) - fs)
    problem.set_performance_metric((x[-1] - x[-2]) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau


def main():
    # test_PEPit_val(10, 1, 1, 1, 1, N=2)
    p = test_admm_pep(10, 2, 1, 1, N=9)
    print(p)


if __name__ == '__main__':
    main()
