import numpy as np
import cvxpy as cp
import scipy.sparse as spa
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction


def test_GD_PEPit():
    problem = PEP()
    mu = 1
    L = 10
    gamma = 1 / L

    func = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    xs = func.stationary_point()
    fs = func.value(xs)

    x0 = problem.set_initial_point()

    problem.set_initial_condition((x0 - xs) ** 2 <= 1)

    # Run n steps of the GD method
    n = 1
    x_iter = [x0]
    x = x0
    for _ in range(n):
        x = x - gamma * func.gradient(x)
        x_iter.append(x)
    print(x_iter)

    # Set the performance metric to the function values accuracy
    # problem.set_performance_metric(func.value(x) - fs)
    problem.set_performance_metric((x_iter[n] - x_iter[n-1]) ** 2)

    # Solve the PEP
    pepit_verbose = 1
    pepit_tau = problem.solve(verbose=pepit_verbose)


def main():
    test_GD_PEPit()


if __name__ == '__main__':
    main()
