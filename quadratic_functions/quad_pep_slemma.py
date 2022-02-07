import numpy as np
import cvxpy as cp


def form_and_solve_sdp():
    n = 3
    mu = 2
    L = 20
    R = 1
    gamma = 2 / (mu + L)
    I = np.eye(n)
    P = np.array([[mu, 0, 0], [0, L, 0], [0, 0, mu]])
    print(P)
    Qb = -(gamma ** 2) * P @ P + 2 * gamma * P - I
    print(Qb)
    ub = np.zeros(n)
    cb = 0

    Qa = -I
    ua = np.zeros(n)
    ca = R ** 2

    eta = cp.Variable()
    lambd = cp.Variable()

    M = cp.Variable((n+1, n+1), symmetric=True)
    constraints = [M >> 0, lambd >= 0]

    constraints.append(M[0:n, 0:n] == Qb - lambd * Qa)
    constraints.append(M[0:n, n] == .5 * (ub - lambd * ua))
    constraints.append(M[n][n] == cb - eta - lambd * ca)

    problem = cp.Problem(cp.Maximize(eta), constraints)
    result = problem.solve()

    print(result)


def main():
    form_and_solve_sdp()


if __name__ == '__main__':
    main()
