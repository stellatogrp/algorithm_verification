import mosek
import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp


def test_with_cvxpy(n, P, q, A, l, u):
    print('--------solving the QP with cvxpy--------')
    x = cp.Variable((n, 1))
    obj = cp.Minimize(.5 * cp.quad_form(x, P) + q.T @ x)
    constraints = [l <= A @ x, A @ x <= u]

    prob = cp.Problem(obj, constraints)
    res = prob.solve()
    print(res)
    # print('x =\n', np.round(x.value, 4))


def test_osqp_admm_onestep_pep():
    m = 10
    n = 20
    R = 10
    In = np.eye(n)
    Im = np.eye(m)

    np.random.seed(0)
    # minimization objective setup
    Phalf = np.random.randn(n, n)
    P = Phalf @ Phalf.T
    # print(P, np.linalg.eigvals(P))
    q = np.random.randn(n)
    q = q.reshape(n, 1)
    # print(q)

    # constraints for minimization obj
    A = np.random.randn(m, n)
    l = -1 * np.ones(m)
    l = l.reshape(m, 1)
    u = 5 * np.ones(m)
    u = u.reshape(m, 1)

    sigma = 1e-4
    rho = 2
    rho_inv = 1 / rho

    test_with_cvxpy(n, P, q, A, l, u)

    # these blocks don't depend on iteration count
    C = P + sigma * In + rho * A.T @ A
    G = np.block([sigma * In, -A.T, rho * A.T])
    H = np.block([rho * A, Im, -rho * Im])
    J = np.block([A, rho_inv * Im])

    # initial iterates
    x0 = cp.Variable((n, 1))
    y0 = cp.Variable((m, 1))
    z0 = cp.Variable((m, 1))

    u0 = cp.vstack([x0, y0, z0])
    x0x0T = cp.Variable((n, n))
    x0y0T = cp.Variable((n, m))
    x0z0T = cp.Variable((n, m))
    y0y0T = cp.Variable((m, m))
    y0z0T = cp.Variable((m, m))
    z0z0T = cp.Variable((m, m))
    u0u0T = cp.bmat([[x0x0T, x0y0T, x0z0T],
                     [x0y0T.T, y0y0T, y0z0T],
                     [x0z0T.T, y0z0T.T, z0z0T]
                     ])

    constraints = [cp.sum_squares(x0) <= R ** 2, cp.trace(x0x0T) <= R ** 2,
                   # cp.reshape(cp.diag(x0x0T), (n, 1)) <= 1,
                   y0 == 0, y0y0T == 0,
                   # cp.sum_squares(y0) <= .1 * R ** 2, cp.trace(y0y0T) <= .1 * R ** 2,
                   z0 == 0, z0z0T == 0]

    # constraints for x^{k+1}
    x1 = cp.Variable((n, 1))
    # y1 = cp.Variable((m, 1))
    # y1y1T = cp.Variable((m, m))
    x1x1T = cp.Variable((n, n))
    x1y0T = cp.Variable((n, m))
    x1z0T = cp.Variable((n, m))
    constraints.append(C @ x1 == G @ u0 - q)
    constraints.append(C @ x1x1T @ C.T == G @ u0u0T @ G.T - G @ u0 @ q.T - q @ u0.T @ G.T + q @ q.T)

    # constraints for y^{k+1}
    v0 = cp.vstack([x1, y0, z0])
    v0v0T = cp.bmat([[x1x1T, x1y0T, x1z0T],
                     [x1y0T.T, y0y0T, y0z0T],
                     [x1z0T.T, y0z0T.T, z0z0T]
                     ])

    y1 = H @ v0
    y1y1T = H @ v0v0T @ H.T

    # constraints for z^{k+1} i.e. box projection
    wtilde0 = cp.vstack([x1, y1])
    x1y1T = cp.bmat([[x1x1T, x1y0T, x1z0T]]) @ H.T
    wtilde0wtilde0T = cp.bmat([[x1x1T, x1y1T],
                               [x1y1T.T, y1y1T]
                               ])
    w0 = J @ wtilde0
    w0w0T = J @ wtilde0wtilde0T @ J.T

    z1 = cp.Variable((m, 1))
    alpha1 = cp.Variable((m, 1))
    z1z1T = cp.Variable((m, m))
    z1alpha1T = cp.Variable((m, m))
    z1w0T = cp.Variable((m, m))
    alpha1alphaT = cp.Variable((m, m))
    alpha1w0T = cp.Variable((m, m))
    P1 = cp.bmat([[z1z1T, z1alpha1T, z1w0T, z1],
                  [z1alpha1T.T, alpha1alphaT, alpha1w0T, alpha1],
                  [z1w0T.T, alpha1w0T.T, w0w0T, w0],
                  [z1.T, alpha1.T, w0.T, np.array([[1]])]
                  ])

    constraints += [alpha1 >= w0, alpha1 >= l,
                    cp.diag(alpha1alphaT - alpha1w0T - l @ alpha1.T + l @ w0.T) == 0,
                    z1 <= alpha1, z1 <= u,
                    cp.diag(u @ alpha1.T - u @ z1.T - z1alpha1T + z1z1T) == 0,
                    # P1 >> 0, u0u0T >> 0, v0v0T >> 0, wtilde0wtilde0T >> 0
                    P1 >> 0,
                    ]
    constraints += [cp.bmat([[u0u0T, u0],
                             [u0.T, np.array([[1]])]]) >> 0]
    constraints += [cp.bmat([[v0v0T, v0],
                             [v0.T, np.array([[1]])]]) >> 0]
    constraints += [cp.bmat([[wtilde0wtilde0T, wtilde0],
                             [wtilde0.T, np.array([[1]])]]) >> 0]

    x1x0T = cp.Variable((n, n))
    constraints.append(C @ x1x0T == sigma * x0x0T - q @ x0.T + rho * A.T @ x0z0T.T - A.T @ x0y0T.T)

    # a few extra variables needed to create the objective
    z1z0T = cp.Variable((m, m))
    y1y0T = cp.Variable((m, m))
    Z = cp.bmat([[z1z1T, z1z0T, z1],
                 [z1z0T.T, z0z0T, z0],
                 [z1.T, z0.T, np.array([[1]])]
                 ])
    Y = cp.bmat([[y1y1T, y1y0T, y1],
                 [y1y0T.T, y0y0T, y0],
                 [y1.T, y0.T, np.array([[1]])]
                 ])
    constraints += [Y >> 0, Z >> 0]

    obj = cp.Maximize(cp.trace(x1x1T - 2 * x1x0T + x0x0T)
                     + cp.trace(y1y1T - 2 * y1y0T + y0y0T)
                     + cp.trace(z1z1T - 2 * z1z0T + z0z0T))
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=True)
    print('sdp result =', res)
    print('obj norm using vectors values =', np.linalg.norm(x1.value-x0.value) ** 2
          + np.linalg.norm(y1.value-y0.value) ** 2
          + np.linalg.norm(z1.value-z0.value) ** 2)


def test_osqp_admm_onestep_pep_mult_moving_parts():
    print('--------solving pep with more moving parts--------')
    m = 5
    n = 10
    R = 2
    In = spa.eye(n)
    # Zmn = np.zeros((m, n))
    Zmn = spa.csc_matrix((m, n))
    Im = spa.eye(m)

    np.random.seed(0)
    Phalf = np.random.randn(n, n)
    P = Phalf @ Phalf.T
    P = spa.csc_matrix(P)
    q_val = np.random.randn(n)
    q_val = q_val.reshape(n, 1)
    # print(q_val)

    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    l = -1 * np.ones(m)
    l = l.reshape(m, 1)
    u = 5 * np.ones(m)
    u = u.reshape(m, 1)

    sigma = 1e-4
    rho = 2
    rho_inv = 1 / rho

    C = spa.bmat([[P + sigma * In + rho * A.T @ A, -rho * A.T, A.T, -sigma * In, In]])
    C = spa.csc_matrix(C)
    H = spa.bmat([[rho * A, -rho * Im, Im, Zmn, Zmn]])
    H = spa.csc_matrix(H)
    J = spa.bmat([[A, rho_inv * Im]])
    J = spa.csc_matrix(J)

    # bounds on q
    q = cp.Variable((n, 1))
    ql = 1
    q2 = 2

    # initial iterates
    x0 = cp.Variable((n, 1))
    y0 = cp.Variable((m, 1))
    z0 = cp.Variable((m, 1))

    # next x iterate
    x1 = cp.Variable((n, 1))

    # outer products for x1 z0 y0 x0 q
    x1x1T = cp.Variable((n, n))
    x1z0T = cp.Variable((n, m))
    x1y0T = cp.Variable((n, m))
    x1x0T = cp.Variable((n, n))
    x1qT = cp.Variable((n, n))

    z0z0T = cp.Variable((m, m))
    z0y0T = cp.Variable((m, m))
    z0x0T = cp.Variable((m, n))
    z0qT = cp.Variable((m, n))

    y0y0T = cp.Variable((m, m))
    y0x0T = cp.Variable((m, n))
    y0qT = cp.Variable((m, n))

    x0x0T = cp.Variable((n, n))
    x0qT = cp.Variable((n, n))

    qqT = cp.Variable((n, n))

    # form v0 and v0v0T for x1 update
    v0 = cp.vstack([x1, z0, y0, x0, q])
    v0v0T = cp.bmat([
        [x1x1T, x1z0T, x1y0T, x1x0T, x1qT],
        [x1z0T.T, z0z0T, z0y0T, z0x0T, z0qT],
        [x1y0T.T, z0y0T.T, y0y0T, y0x0T, y0qT],
        [x1x0T.T, z0x0T.T, y0x0T.T, x0x0T, x0qT],
        [x1qT.T, z0qT.T, y0qT.T, x0qT.T, qqT]
    ])

    # initial constraints
    constraints = [cp.sum_squares(x0) <= R ** 2, cp.trace(x0x0T) <= R ** 2,
                   # cp.reshape(cp.diag(x0x0T), (n, 1)) <= 1,
                   y0 == 0, y0y0T == 0,
                   # cp.sum_squares(y0) <= .1 * R ** 2, cp.trace(y0y0T) <= .1 * R ** 2,
                   z0 == 0, z0z0T == 0,
                   q == q_val, qqT == np.outer(q_val, q_val),
                   # cp.sum_squares(q) <= .1 * R ** 2, cp.trace(qqT) <= .1 * R ** 2
                   ]

    # constraints to create x1
    constraints += [C @ v0 == 0, C @ v0v0T @ C.T == 0,
                    cp.bmat([
                        [v0v0T, v0],
                        [v0.T, np.array([[1]])]
                    ]) >> 0]

    y1 = H @ v0
    y1y1T = H @ v0v0T @ H.T

    # constraints for z^{k+1} i.e. box projection
    wtilde0 = cp.vstack([x1, y1])
    # x1y1T = cp.bmat([[x1x1T, x1y0T, x1z0T]]) @ H.T
    x1y1T = cp.Variable((n, m))  # TODO: check this
    wtilde0wtilde0T = cp.bmat([[x1x1T, x1y1T],
                               [x1y1T.T, y1y1T]
                               ])
    constraints += [cp.bmat([[wtilde0wtilde0T, wtilde0],
                             [wtilde0.T, np.array([[1]])]]) >> 0]

    w0 = J @ wtilde0
    w0w0T = J @ wtilde0wtilde0T @ J.T

    z1 = cp.Variable((m, 1))
    alpha1 = cp.Variable((m, 1))
    z1z1T = cp.Variable((m, m))
    z1alpha1T = cp.Variable((m, m))
    z1w0T = cp.Variable((m, m))
    alpha1alphaT = cp.Variable((m, m))
    alpha1w0T = cp.Variable((m, m))
    P1 = cp.bmat([[z1z1T, z1alpha1T, z1w0T, z1],
                  [z1alpha1T.T, alpha1alphaT, alpha1w0T, alpha1],
                  [z1w0T.T, alpha1w0T.T, w0w0T, w0],
                  [z1.T, alpha1.T, w0.T, np.array([[1]])]
                  ])

    constraints += [alpha1 >= w0, alpha1 >= l,
                    cp.diag(alpha1alphaT - alpha1w0T - l @ alpha1.T + l @ w0.T) == 0,
                    z1 <= alpha1, z1 <= u,
                    cp.diag(u @ alpha1.T - u @ z1.T - z1alpha1T + z1z1T) == 0,
                    # P1 >> 0, u0u0T >> 0, v0v0T >> 0, wtilde0wtilde0T >> 0
                    P1 >> 0,
                    ]

    z1z0T = cp.Variable((m, m))
    y1y0T = cp.Variable((m, m))
    Z = cp.bmat([[z1z1T, z1z0T, z1],
                 [z1z0T.T, z0z0T, z0],
                 [z1.T, z0.T, np.array([[1]])]
                 ])
    Y = cp.bmat([[y1y1T, y1y0T, y1],
                 [y1y0T.T, y0y0T, y0],
                 [y1.T, y0.T, np.array([[1]])]
                 ])
    constraints += [Y >> 0, Z >> 0]

    obj = cp.Maximize(cp.trace(x1x1T - 2 * x1x0T + x0x0T)
                      + cp.trace(y1y1T - 2 * y1y0T + y0y0T)
                      + cp.trace(z1z1T - 2 * z1z0T + z0z0T))
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=True)
    print('sdp result =', res)


def main():
    # test_osqp_admm_onestep_pep()
    test_osqp_admm_onestep_pep_mult_moving_parts()


if __name__ == '__main__':
    main()
