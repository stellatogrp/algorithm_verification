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

    problem.set_initial_condition((x0 - xs) ** 2 <= 25)

    # Run N steps of the GD method
    N = 5
    x_iter = [x0]
    x = x0
    for _ in range(N):
        x = x - gamma * func.gradient(x)
        x_iter.append(x)
    print(x_iter)

    # Set the performance metric to the function values accuracy
    # problem.set_performance_metric(func.value(x) - fs)
    problem.set_performance_metric((x_iter[N] - x_iter[N-1]) ** 2)

    # Solve the PEP
    pepit_verbose = 1
    pepit_tau = problem.solve(verbose=pepit_verbose)


def test_GD_SDR():
    N = 5
    n = 2
    t = .05
    R = 5
    In = spa.eye(n)

    mu = 1
    L = 10

    np.random.seed(0)
    # Phalf = np.random.randn(n, n)
    # P = Phalf @ Phalf.T
    P = 2 * np.array([[mu, 0], [0, L]])

    q = cp.Variable((n, 1))
    qqT = cp.Variable((n, n))
    x0 = cp.Variable((n, 1))
    x0x0T = cp.Variable((n, n))

    constraints = [cp.sum_squares(x0) <= R ** 2, cp.trace(x0x0T) <= R ** 2,
                   cp.sum_squares(q) <= .5 * R ** 2, cp.trace(qqT) <= .5 * R ** 2]

    C = spa.bmat([[In, t * P - In, t * In]])

    x_vars = [x0]
    xxT_vars = [x0x0T]
    xcross_vars = [0]

    xk = x0
    xk_xkT = x0x0T
    xk_qT = cp.Variable((n, n))
    for k in range(N):
        xkplus1 = cp.Variable((n, 1))
        xkplus1_xkplus1T = cp.Variable((n, n))

        u = cp.vstack([xkplus1, xk, q])

        xkplus1_qT = cp.Variable((n, n))
        xkplus1_xkT = cp.Variable((n, n))

        uuT = cp.bmat([
            [xkplus1_xkplus1T, xkplus1_xkT, xkplus1_qT],
            [xkplus1_xkT.T, xk_xkT, xk_qT],
            [xkplus1_qT.T, xk_qT.T, qqT],
        ])

        constraints += [
            C @ u == 0, C @ uuT @ C.T == 0,
            cp.bmat([
                [uuT, u],
                [u.T, np.array([[1]])]
            ]) >> 0,
        ]

        x_vars.append(xkplus1)
        xxT_vars.append(xkplus1_xkplus1T)
        xcross_vars.append(xkplus1_xkT)

        xk = xkplus1
        xk_xkT = xkplus1_xkplus1T
        xk_qT = xkplus1_qT
    print(len(x_vars), len(xxT_vars), len(xcross_vars))
    obj = cp.Maximize(cp.trace(xxT_vars[-1]) - 2 * cp.trace(xcross_vars[-1]) + cp.trace(xxT_vars[-2]))
    problem = cp.Problem(obj, constraints)
    res = problem.solve(solver=cp.MOSEK, verbose=False)
    print(res)


def main():
    test_GD_PEPit()
    test_GD_SDR()


if __name__ == '__main__':
    main()
