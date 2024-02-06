import cvxpy as cp
import numpy as np


def dd_constraints(D):
    return [2 * cp.diag(D) >= cp.sum(cp.abs(D), axis=1)]


def cauchy_constraints(X):
    n = X.shape[0]
    constraints = []
    for i in range(0, n):
        for j in range(i+1, n):
            constraints += [cp.abs(X[i][j]) <= (X[i][i] + X[j][j]) / 2]
    return constraints


def solve_dd_cvxpy(C, A_vals, b_lvals, b_uvals, PSD_cones, problem_dim):
    print('----solving dd relaxation----')
    X = cp.Variable((problem_dim, problem_dim), symmetric=True)
    # X = cp.Variable((problem_dim, problem_dim))
    obj = cp.trace(-C @ X)

    constraints = []
    for Ai, bli, bui in zip(A_vals, b_lvals, b_uvals):
        if bli > -np.inf:
            constraints += [cp.trace(Ai @ X) >= bli]
        if bui < np.inf:
            constraints += [cp.trace(Ai @ X) <= bui]
    # constraints += [X >> 0]
    # constraints += dd_constraints(X)
    constraints += cauchy_constraints(X)

    prob = cp.Problem(cp.Maximize(obj), constraints)
    res = prob.solve(verbose=True)
    print(res)

    exit(0)
    return 0
