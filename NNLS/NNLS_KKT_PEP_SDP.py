import numpy as np
import cvxpy as cp
import scipy.sparse as spa


def test_NNLS_KKT_PEP_SDP():
    np.random.seed(0)

    n = 2
    m = 3
    t = .05
    R = 1

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    I = np.eye(n)
    halfATA = .5 * A.T @ A
    bTA = b.T @ A
    tbTA = t * b.T @ A
    tATb = t * A.T @ b
    C = I - t * A.T @ A

    print(A, b)

    print('--------testing n=%d --------' % n)
    P = cp.Variable((3 * n + 1, 3 * n + 1), symmetric=True)
    l = -1
    u = 1

    # P = v v.T , where v = [x1 gamma x0 1] stack
    # x0 is initial point, x1 is final point, gamma is lagrange multiplier
    obj = cp.trace(halfATA @ P[0: n, 0: n]) - bTA @ P[0:n, -1] + .5 * b.T @ b
    constraints = [P >> 0, P[-1, -1] == 1]
    constraints.append(obj <= 20)

    # initial bounds
    # constraints.append(cp.diag(P[2 * n: 3 * n, 2 * n: 3 * n]) <= (l + u) * P[2 * n: 3 * n, -1] - l * u)
    constraints.append(cp.trace(P[2 * n: 3 * n, 2 * n: 3 * n]) <= R ** 2)
    constraints.append(cp.sum_squares(P[2 * n: 3 * n, -1]) <= R ** 2)

    # bounds on x1
    constraints.append(P[0: n, -1] >= 0)
    constraints.append(P[0: n, 0: n] >= 0)

    # bounds on gamma
    constraints.append(P[n: 2 * n, -1] >= 0)
    constraints.append(P[n: 2 * n, n: 2 * n] >= 0)

    # slackness constraint
    # constraints.append()
    constraints.append(P[0: n, n: 2 * n] == 0)

    # GD step bounds for x1 and x1 x1.T
    constraints.append(P[0: n, -1] == C @ P[2 * n: 3 * n, -1] + P[n: 2 * n, -1] + tATb)
    constraints.append(P[0: n, 0: n]
                       == C @ P[2 * n: 3 * n, 2 * n: 3 * n] @ C.T + C @ P[n: 2 * n, 2 * n: 3 * n].T
                       + C @ cp.reshape(P[2 * n: 3 * n, -1], (n, 1)) @ tATb.reshape(1, n)
                       + P[n: 2 * n, 2 * n: 3 * n] @ C.T + P[n: 2 * n, n: 2 * n]
                       + cp.reshape(P[n: 2 * n, -1], (n, 1)) @ tATb.reshape(1, n)
                       + tATb.reshape(n, 1) @ cp.reshape(P[2 * n: 3 * n, -1], (1, n)) @ C.T
                       + tATb.reshape(n, 1) @ cp.reshape(P[n: 2 * n, -1], (1, n)) + np.outer(tATb, tATb))

    problem = cp.Problem(cp.Maximize(obj), constraints)
    result = problem.solve(solver=cp.MOSEK, verbose=True)
    print(result)
    print(np.round(P.value, 4))

    
def main():
    test_NNLS_KKT_PEP_SDP()


if __name__ == '__main__':
    main()
