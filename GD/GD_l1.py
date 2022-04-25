import numpy as np
import cvxpy as cp
import scipy.sparse as spa


def test_l1_SDR():
    R = 1
    n = 3

    np.random.seed(0)
    A = np.random.randn(n, n)
    c = np.zeros(n).reshape(n, 1)
    # c = -np.ones(n).reshape(n, 1)
    ccT = np.outer(c, c)
    ones = np.ones(n).reshape(n, 1)

    x = cp.Variable((n, 1))
    xxT = cp.Variable((n, n), symmetric=True)
    u = cp.Variable((n, 1))
    uuT = cp.Variable((n, n), symmetric=True)

    constraints = [x - c <= u, -(x - c) <= u, uuT >= 0]
    constraints += [xxT - x @ c.T - c @ x.T + c @ c.T <= uuT, ones.T @ u <= R, ones.T @ uuT @ ones <= R ** 2]

    constraints += [
        cp.bmat([
            [xxT, x],
            [x.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [uuT, u],
            [u.T, np.array([[1]])]
        ]) >> 0
    ]

    obj = cp.Maximize(cp.trace(A @ uuT))
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.MOSEK)
    print(res)
    print(np.round(x.value, 4))
    print(np.round(u.value, 4))


def test_GD_l1_SDR():
    R = 1
    n = 3


def main():
    test_l1_SDR()


if __name__ == '__main__':
    main()
