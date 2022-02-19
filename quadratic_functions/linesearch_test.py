import numpy as np
import cvxpy as cp


def test_linesearch_sdp():
    n = 2
    mu = 2
    L = 20
    R = 1
    gamma = 2 / (mu + L)
    I = np.eye(n)
    Z = np.zeros((n, n))
    P = np.array([[mu, 0, 0], [0, L, 0], [0, 0, (mu + L) / 2]])
    P = np.array([[mu, 0], [0, L]])

    Hobj = -.5 * np.block([[Z, Z], [Z, P]])
    cobj = np.zeros(2 * n)
    dobj = 0
    print(P, Hobj)

    # Hineq = np.block([[I, Z], [Z, Z]])
    Hineq = .5 * np.block([[P, Z], [Z, Z]])
    cineq = np.zeros(2 * n)
    dineq = -R ** 2

    Heq1 = np.block([[Z, -.5 * P], [-.5 * P.T, P]])
    ceq1 = np.zeros(2 * n)
    deq1 = 0
    # print(Heq1)

    Heq2 = np.block([[Z, .5 * P @ P], [.5 * P @ P, Z]])
    ceq2 = np.zeros(2 * n)
    deq2 = 0
    # print(Heq2)

    M = cp.Variable((2 * n + 1, 2 * n + 1), symmetric=True)
    eta = cp.Variable()
    lambd = cp.Variable()
    kappa1 = cp.Variable()
    kappa2 = cp.Variable()

    constraints = [M >> 0, lambd >= 0]

    constraints.append(M[0:2 * n, 0:2 * n] == Hobj + lambd * Hineq - kappa1 * Heq1 - kappa2 * Heq2)
    constraints.append(M[0:2 * n, 2 * n] == .5 * (cobj + lambd * cineq - kappa1 * ceq1 - kappa2 * ceq2))
    constraints.append(M[2 * n][2 * n] == dobj - eta + lambd * dineq - kappa1 * deq1 - kappa2 * deq2)

    problem = cp.Problem(cp.Maximize(eta), constraints)
    result = problem.solve()

    print('sdp result:', result)
    for i in [0]:
        print('constraint', i)
        dual = constraints[i].dual_value
        print('dual var', dual)
        dual[dual < 1e-6] = 0
        print(dual)

    test = constraints[i].dual_value[0:2 * n, 0:2 * n]
    test[np.abs(test) < 1e-8] = 0
    print(test)

    # U, sigma, VT = np.linalg.svd(test, full_matrices=False)
    # print(sigma)
    # print(VT[0])
    # print(sigma[0] * np.outer(VT[0], VT[0]))

    print('testing alternative SDP relaxation')
    X = cp.Variable((2 * n, 2 * n))
    x = cp.Variable(2 * n)
    N = cp.Variable((2 * n + 1, 2 * n + 1), symmetric=True)

    obj = cp.trace(Hobj @ X) + 2 * cobj @ x + dobj
    constraints = [N >> 0, N[0:2 * n, 0:2 * n] == X, N[0:2 * n, 2 * n] == x, N[2 * n][2 * n] == 1]
    constraints.append(N[2 * n][0] == 1 / mu)
    constraints.append(N[2 * n][1] == 1 / L)

    constraints.append(cp.trace(Hineq @ X) + 2 * cineq @ x + dineq <= 0)
    constraints.append(cp.trace(Heq1 @ X) + 2 * ceq1 @ x + deq1 == 0)
    constraints.append(cp.trace(Heq2 @ X) + 2 * ceq2 @ x + deq2 == 0)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    result = problem.solve()
    print('alt result', result)
    print(N.value)


def main():
    test_linesearch_sdp()


if __name__ == '__main__':
    main()
