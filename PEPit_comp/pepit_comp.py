import numpy as np
import matplotlib.pyplot as plt
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from quadratic_functions.extended_slemma_sdp import *


def generate_initial_point(n):
    random_point = np.random.normal(0, 1, n)
    x_center = random_point / np.linalg.norm(random_point)
    print('center:', x_center, 'norm:', np.linalg.norm(x_center))
    return x_center


def main():
    n = 2
    mu = 1
    L = 10
    gamma = 2 / (mu + L)
    P = np.array([[mu, 0], [0, L]])
    epsilon = .1

    np.random.seed(1)
    initial_point = generate_initial_point(n)

    R_vals = np.array([1, 2, 3, 4, 5])
    pepit_vals = test_PEPit_vals(mu, L, gamma, R_vals + epsilon)
    off_center_vals = test_off_center_vals(n, P, gamma, epsilon, initial_point, R_vals)

    fig, ax = plt.subplots()

    plt.plot(R_vals, pepit_vals, 'go', label='PEPit')
    plt.plot(R_vals, off_center_vals, 'ro', label='off center')

    plt.title(f'x0 = {np.round(initial_point, 3)}')
    plt.xlabel('R')
    plt.ylabel('f(x^N) - f(x^\star)')

    plt.legend()

    plt.show()


def test_PEPit_vals(mu, L, gamma, R_vals, N=1):
    pepit_tau_vals = []
    for R in R_vals:
        problem = PEP()
        func = problem.declare_function(SmoothStronglyConvexFunction, param={'L': L, 'mu': mu})

        # Start by defining its unique optimal point xs = x_* and corresponding function value fs = f_*
        xs = func.stationary_point()
        fs = func.value(xs)

        # Then define the starting point x0 of the algorithm
        x0 = problem.set_initial_point()

        # Set the initial constraint that is the distance between x0 and x^*
        problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)

        # Run n steps of the GD method
        x = x0
        for _ in range(N):
            x = x - gamma * func.gradient(x)

        # Set the performance metric to the function values accuracy
        problem.set_performance_metric(func.value(x) - fs)

        # Solve the PEP
        pepit_verbose = 0
        pepit_tau = problem.solve(verbose=pepit_verbose)

        pepit_tau_vals.append(pepit_tau)
    print(pepit_tau_vals)
    return pepit_tau_vals


def test_off_center_vals(n, P, gamma, epsilon, initial_point, R_vals):
    off_center_point_vals = []

    Hobj = .5 * (-(gamma ** 2) * P @ P @ P + 2 * gamma * P @ P - P)
    cobj = np.zeros(n)
    dobj = 0

    Hineq = np.eye(n)
    for R in R_vals:
        initial_point = initial_point / np.linalg.norm(initial_point)
        initial_point = initial_point * R

        cineq = -2 * initial_point
        dineq = -epsilon ** 2 + np.inner(initial_point, initial_point)

        result, N = solve_full_extended_slemma_primal_sdp(n, (Hobj, cobj, dobj), ineq_param_lists=[(Hineq, cineq, dineq)],
                                                      verbose=False)
        off_center_point_vals.append(-result)
    print(off_center_point_vals)
    return off_center_point_vals


if __name__ == '__main__':
    main()