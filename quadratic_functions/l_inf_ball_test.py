import numpy as np
import cvxpy as cp


def test_hardcoded_l_inf_sdp():
    n = 3
    mu = 2
    L = 20
    R = 1
    gamma = 2 / (mu + L)
    I = np.eye(n)
    P = np.array([[mu, 0, 0], [0, L, 0], [0, 0, 5]])
    Qb = -(gamma ** 2) * P @ P + 2 * gamma * P - I
    ub = np.zeros(n)
    cb = 0

    Qa = np.outer(I[0], I[0])
    # Qa = I
    ua = np.zeros(n)
    ca = -R ** 2

    Qa2 = np.outer(I[1], I[1])
    ua2 = np.zeros(n)
    ca2 = -R ** 2

    Qa3 = np.outer(I[2], I[2])
    ua3 = np.zeros(n)
    ca3 = -R ** 2

    eta = cp.Variable()
    lambd = cp.Variable()
    lambd2 = cp.Variable()
    lambd3 = cp.Variable()

    M = cp.Variable((n + 1, n + 1), symmetric=True)
    constraints = [M >> 0, lambd >= 0, lambd2 >= 0]

    constraints.append(M[0:n, 0:n] == Qb + lambd * Qa + lambd2 * Qa2 + lambd3 * Qa3)
    constraints.append(M[0:n, n] == .5 * (ub + lambd * ua + lambd2 * ua2 + lambd3 * ua3))
    constraints.append(M[n][n] == cb - eta + lambd * ca + lambd2 * ca2 + lambd3 * ca3)

    problem = cp.Problem(cp.Maximize(eta), constraints)
    result = problem.solve()

    print('sdp result:', result)


def main():
    test_hardcoded_l_inf_sdp()


if __name__ == '__main__':
    main()
