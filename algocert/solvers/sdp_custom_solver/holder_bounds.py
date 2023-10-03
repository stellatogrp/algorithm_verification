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
        cp.norm(x1 - x0, p) <= r,
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
        print(min_res, max_res)

        l_out[i, 0] = min_res
        u_out[i, 0] = max_res
    return l_out, u_out


def test_double_holder_cvxpy(p1, c1, r1, A1, b1, p2, c2, r2, A2, b2):
    m, n = A1.shape
    l_out = np.zeros((m, 1))
    u_out = np.zeros((m, 1))
    x1 = cp.Variable((n, 1))
    x2 = cp.Variable((n, 1))
    constraints = [
        cp.norm(x1 - c1, p1) <= r1,
        cp.norm(x2 - c2, p2) <= r2,
    ]
    # Ax1plusb = A1 @ x1 + b1
    sum_var = A1 @ x1 + b1 + A2 @ x2 + b2
    for i in range(m):
        ei = np.zeros((m, 1))
        ei[i, 0] = 1
        # obj = ei.T @ Ax1plusb
        obj = ei.T @ sum_var
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
    # A = np.eye(m)
    b = np.random.randn(m, 1)
    # b = np.zeros((m, 1))
    r = 2
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


def test_double_holder(m, n):
    np.random.seed(0)
    print(m, n)
    A1 = np.random.randn(m, n)
    b1 = np.random.randn(m, 1)
    x1 = np.random.randn(n, 1)
    r1 = 1
    p1 = 2

    A2 = np.random.randn(m, n)
    b2 = np.random.randn(m, 1)
    x2 = np.random.randn(n, 1)
    r2 = 2
    p2 = 2

    l1, u1 = holder_bounds(p1, x1, r1, A1, b1)
    l2, u2 = holder_bounds(p2, x2, r2, A2, b2)
    ladd = l1 + l2
    uadd = u1 + u2

    # print(l1, u1)
    # print(test_double_holder_cvxpy(p, x1, r1, A1, b1, x2, r2, A2, b2))
    l_cp, u_cp = test_double_holder_cvxpy(p1, x1, r1, A1, b1, p2, x2, r2, A2, b2)
    print(l_cp, u_cp)
    print('l_diff:', np.linalg.norm(ladd - l_cp))
    print('u_diff:', np.linalg.norm(uadd - u_cp))


def main():
    m = 5
    n = 3
    # test_holder(m, n)
    test_double_holder(m, n)


if __name__ == '__main__':
    main()
