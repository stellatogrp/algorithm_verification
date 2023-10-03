import cvxpy as cp
import numpy as np


def dual_norm(p):
    if p == 1:
        return np.inf
    if p == np.inf:
        return 1
    return p / (p - 1)


def holder_bounds(p, x0, r, Al, bl, Au=None, bu=None):
    # if Au, bu not given, assume same matrices
    if Au is None:
        Au = Al
    if bu is None:
        bu = bl

    q = dual_norm(p)

    l_h = -r * np.linalg.norm(Al, ord=q, axis=1).reshape((-1, 1)) + Al @ x0 + bl
    u_h = r * np.linalg.norm(Au, ord=q, axis=1).reshape((-1, 1)) + Au @ x0 + bu

    return l_h, u_h


def test_with_cvxpy(p, x0, r, A, b):
    m, n = A.shape
    l_out = np.zeros((m, 1))
    u_out = np.zeros((m, 1))
    x1 = cp.Variable((n, 1))
    constraints = [
        cp.norm(x1 - x0, p) <= r ** 2,
    ]
    Ax1plusb = A @ x1 + b
    for i in range(m):
        ei = np.zeros((m, 1))
        ei[i, 0] = 1
        obj = ei.T @ Ax1plusb
        prob = cp.Problem(cp.Minimize(obj), constraints)
        min_res = prob.solve()

        prob = cp.Problem(cp.Maximize(obj), constraints)
        max_res = prob.solve()
        # print(min_res, max_res)

        l_out[i, 0] = min_res
        u_out[i, 0] = max_res
    return l_out, u_out


def test_holder(m, n):
    np.random.seed(0)
    print(m, n)
    A = np.random.randn(m, n)
    b = np.random.randn(m, 1)
    r = 1
    p_vals = [1, 2, np.inf]
    x0 = np.random.randn(n, 1)
    print(A, b, x0)
    for p in p_vals:
        print('testing p:', p)
        l_cp, u_cp = test_with_cvxpy(p, x0, r, A, b)
        # print(l_cp, u_cp)
        l_h, u_h = holder_bounds(p, x0, r, A, b)
        # print(l_h, u_h)
        print('l_h diff:', np.linalg.norm(l_h - l_cp))
        print('u_h diff:', np.linalg.norm(u_h - u_cp))

def main():
    m = 5
    n = 3
    test_holder(m, n)


if __name__ == '__main__':
    main()
