import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp

from gurobipy import GRB


def test_Nstep_OSQP_SDR():
    N = 1
    m = 5
    n = 3
    R = 2
    In = spa.eye(n)
    # Zmn = np.zeros((m, n))
    Zmn = spa.csc_matrix((m, n))
    Zmm = spa.csc_matrix((m, m))
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
    J = spa.bmat([[A, rho_inv * Im, Zmm]])
    J = spa.csc_matrix(J)

    s_vars = []
    ssT_vars = []

    x0 = cp.Variable((n, 1))
    x0x0T = cp.Variable((n, n))

    y0 = cp.Variable((n, 1))
    y0y0T = cp.Variable((n, n))

    v0 = cp.Variable((n, 1))
    v0v0T = cp.Variable((n, n))

    w0 = cp.Variable((n, 1))
    w0w0T = cp.Variable((n, n))

    b = cp.Variable((m, 1))
    bbT = cp.Variable((m, m))


def test_Nstep_quad_boxproj():
    N = 4
    m = 5
    n = 3
    R = 20
    In = spa.eye(n)
    t = .05
    # Zmn = np.zeros((m, n))
    Zn = spa.csc_matrix((n, n))
    Zmn = spa.csc_matrix((m, n))
    Zm = spa.csc_matrix((m, m))
    Im = spa.eye(m)

    l = -1 * np.ones(n)
    l = l.reshape(n, 1)
    u = 5 * np.ones(n)
    u = u.reshape(n, 1)

    np.random.seed(0)
    Phalf = np.random.randn(n, n)
    P = Phalf @ Phalf.T
    P = spa.csc_matrix(P)
    q_val = np.random.randn(n)
    q_val = q_val.reshape(n, 1)

    x0 = cp.Variable((n, 1))
    x0x0T = cp.Variable((n, n))

    w0 = cp.Variable((n, 1))
    w0w0T = cp.Variable((n, n))

    y0 = cp.Variable((n, 1))
    y0y0T = cp.Variable((n, n))

    q = cp.Variable((n, 1))
    qqT = cp.Variable((n, n))

    constraints = [
        cp.sum_squares(x0) <= R ** 2, cp.trace(x0x0T) <= R ** 2,
        # x0 == 0, x0x0T == 0,
        y0 == 0, y0y0T == 0,
        w0 == 0, w0w0T == 0,
        cp.sum_squares(q) <= .5 * R ** 2, cp.trace(qqT) <= .5 * R ** 2,
        # b == 0, bbT == 0,
    ]

    x0w0T = cp.Variable((n, n))
    x0y0T = cp.Variable((n, n))
    x0qT = cp.Variable((n, n))

    w0y0T = cp.Variable((n, n))
    w0qT = cp.Variable((n, n))

    y0qT = cp.Variable((n, n))

    s0 = cp.vstack([x0, w0, y0])
    s0s0T = cp.bmat([
        [x0x0T, x0w0T, x0y0T],
        [x0w0T.T, w0w0T, w0y0T],
        [x0y0T.T, w0y0T.T, y0y0T]
    ])

    C = spa.bmat([[Zn, Zn, In, t * P - In, Zn, Zn, t * In]])

    s_vars = [s0]
    ssT_vars = [s0s0T]
    scross_vars = [0]

    state_n = s0.shape[0]
    # print(state_n)
    sk = s0
    sk_skT = s0s0T
    sk_qT = cp.vstack([x0qT, w0qT, y0qT])
    for k in range(N):
        xkplus1 = cp.Variable((n, 1))
        xkplus1_xkplus1T = cp.Variable((n, n))

        wkplus1 = cp.Variable((n, 1))
        wkplus1_wkplus1T = cp.Variable((n, n))

        ykplus1 = cp.Variable((n, 1))
        ykplus1_ykplus1T = cp.Variable((n, n))

        xkplus1_wkplus1T = cp.Variable((n, n))
        xkplus1_ykplus1T = cp.Variable((n, n))
        xkplus1_qT = cp.Variable((n, n))

        wkplus1_ykplus1T = cp.Variable((n, n))
        wkplus1_qT = cp.Variable((n, n))

        ykplus1_qT = cp.Variable((n, n))

        skplus1 = cp.vstack([xkplus1, wkplus1, ykplus1])
        skplus1_skplus1T = cp.bmat([
            [xkplus1_xkplus1T, xkplus1_wkplus1T, xkplus1_ykplus1T],
            [xkplus1_wkplus1T.T, wkplus1_wkplus1T, wkplus1_ykplus1T],
            [xkplus1_ykplus1T.T, wkplus1_ykplus1T.T, ykplus1_ykplus1T]
        ])
        skplus1_qT = cp.vstack([xkplus1_qT, wkplus1_qT, ykplus1_qT])
        skplus1_skT = cp.Variable((state_n, state_n))

        # print(skplus1, sk, q)
        v = cp.vstack([skplus1, sk, q])

        vvT = cp.bmat([
            [skplus1_skplus1T, skplus1_skT, skplus1_qT],
            [skplus1_skT.T, sk_skT, sk_qT],
            [skplus1_qT.T, sk_qT.T, qqT]
        ])

        # constraints
        constraints += [
            C @ v == 0, C @ vvT @ C.T == 0,
            cp.bmat([
                [vvT, v],
                [v.T, np.array([[1]])]
            ]) >> 0,
            # xkplus1 >= l, xkplus1 >= ykplus1,
            # cp.diag(xkplus1_xkplus1T - xkplus1_ykplus1T - l @ ykplus1.T + l @ xkplus1.T) == 0,
            wkplus1 >= l, wkplus1 >= ykplus1,
            cp.diag(wkplus1_wkplus1T) >= cp.diag(l @ l.T), cp.diag(wkplus1_wkplus1T) >= cp.diag(ykplus1_ykplus1T),
            cp.diag(wkplus1_wkplus1T - wkplus1_ykplus1T - l @ ykplus1.T + l @ wkplus1.T) == 0,
            xkplus1 <= u, xkplus1 <= wkplus1,
            cp.diag(xkplus1_xkplus1T) <= cp.diag(u @ u.T), cp.diag(xkplus1_xkplus1T) <= cp.diag(wkplus1_wkplus1T),
            cp.diag(u @ wkplus1.T - u @ xkplus1.T - xkplus1_wkplus1T + xkplus1_xkplus1T) == 0,
        ]

        s_vars.append(skplus1)
        ssT_vars.append(skplus1_skplus1T)
        scross_vars.append(skplus1_skT)

        sk = skplus1
        sk_skT = skplus1_skplus1T
        sk_qT = skplus1_qT

    print(len(s_vars), len(ssT_vars), len(scross_vars))
    sN_sNT = ssT_vars[-1]
    sNminus1_sNminus1T = ssT_vars[-1]
    scross = scross_vars[-1]

    xN_xNT = sN_sNT[0: n, 0: n]
    xNminus1_xNminus1T = sNminus1_sNminus1T[0: n, 0: n]
    xN_xNminus1T = scross[0: n, 0: n]

    obj = cp.Maximize(cp.trace(xN_xNT - 2 * xN_xNminus1T + xNminus1_xNminus1T))
    problem = cp.Problem(obj, constraints)
    res = problem.solve(solver=cp.MOSEK, verbose=True)
    print(res)


def test_Nstep_OSQP_SDR_alt():
    N = 1
    m = 10
    n = 5
    R = 20
    In = spa.eye(n)
    # Zmn = np.zeros((m, n))
    Zmn = spa.csc_matrix((m, n))
    Zmm = spa.csc_matrix((m, m))
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
    J = spa.bmat([[A, rho_inv * Im, Zmm]])
    J = spa.csc_matrix(J)

    x_vars = []
    xxT_vars = []
    y_vars = []
    yyT_vars = []
    z_vars = []
    zzT_vars = []

    # initial iterates
    xk = cp.Variable((n, 1))
    xk_xkT = cp.Variable((n, n))
    x_vars.append(xk)
    xxT_vars.append(xk_xkT)

    yk = cp.Variable((m, 1))
    yk_ykT = cp.Variable((m, m))
    y_vars.append(yk)
    yyT_vars.append(yk_ykT)

    zk = cp.Variable((m, 1))
    zk_zkT = cp.Variable((m, m))
    z_vars.append(zk)
    zzT_vars.append(zk_zkT)

    q = cp.Variable((n, 1))
    qqT = cp.Variable((n, n))
    ql = 1
    q2 = 2

    zk_ykT = cp.Variable((m, m))
    zk_xkT = cp.Variable((m, n))
    yk_xkT = cp.Variable((m, n))

    zk_qT = cp.Variable((m, n))
    yk_qT = cp.Variable((m, n))
    xk_qT = cp.Variable((n, n))

    constraints = [cp.sum_squares(xk) <= R ** 2, cp.trace(xk_xkT) <= R ** 2,
                   yk == 0, yk_ykT == 0,
                   # cp.sum_squares(yk) <= R ** 2, cp.trace(yk_ykT) <= R ** 2,
                   zk == 0, zk_zkT == 0,
                   # zk == A @ xk, zk_zkT == A @ xk_xkT @ A.T, zk_xkT == A @ xk_xkT,
                   q == q_val, qqT == np.outer(q_val, q_val),
                   # cp.sum_squares(q) <= .2 * R ** 2, cp.trace(qqT) <= .2 * R ** 2,
                   ]

    for _ in range(N):
        xkplus1 = cp.Variable((n, 1))
        xkplus1_xkplus1T = cp.Variable((n, n))
        v = cp.vstack([xkplus1, zk, yk, xk, q])

        xkplus1_zkT = cp.Variable((n, m))
        xkplus1_ykT = cp.Variable((n, m))
        xkplus1_xkT = cp.Variable((n, n))
        xkplus1_qT = cp.Variable((n, n))

        # these six need to carry over to next iteration
        # zk_ykT = cp.Variable((m, m))
        # zk_xkT = cp.Variable((m, n))
        # yk_xkT = cp.Variable((m, n))

        zk_qT = cp.Variable((m, n))
        yk_qT = cp.Variable((m, n))
        # xk_qT = cp.Variable((n, n))

        vvT = cp.bmat([
            [xkplus1_xkplus1T, xkplus1_zkT, xkplus1_ykT, xkplus1_xkT, xkplus1_qT],
            [xkplus1_zkT.T, zk_zkT, zk_ykT, zk_xkT, zk_qT],
            [xkplus1_ykT.T, zk_ykT.T, yk_ykT, yk_xkT, yk_qT],
            [xkplus1_xkT.T, zk_xkT.T, yk_xkT.T, xk_xkT, xk_qT],
            [xkplus1_qT.T, zk_qT.T, yk_qT.T, xk_qT.T, qqT]
        ])

        constraints += [C @ v == 0, C @ vvT @ C.T == 0,
            cp.bmat([
                [vvT, v],
                [v.T, np.array([[1]])]
            ]) >> 0,
        ]

        ykplus1 = H @ v
        ykplus1_ykplus1T = H @ vvT @ H.T

        zkplus1 = cp.Variable((m, 1))
        zkplus1_zkplus1T = cp.Variable((m, m))

        wtilde = cp.vstack([xkplus1, ykplus1, zkplus1])
        xkplus1_ykplus1T = cp.Variable((n, m))
        xkplus1_zkplus1T = cp.Variable((n, m))
        ykplus1_zkplus1T = cp.Variable((m, m))
        wtilde_wtildeT = cp.bmat([
            [xkplus1_xkplus1T, xkplus1_ykplus1T, xkplus1_zkplus1T],
            [xkplus1_ykplus1T.T, ykplus1_ykplus1T, ykplus1_zkplus1T],
            [xkplus1_zkplus1T.T, ykplus1_zkplus1T.T, zkplus1_zkplus1T]
        ])
        constraints += [
            cp.bmat([
                [wtilde_wtildeT, wtilde],
                [wtilde.T, np.array([[1]])]
            ]) >> 0
        ]

        w = J @ wtilde
        wwT = J @ wtilde_wtildeT @ J.T


        ztilde = cp.Variable((m, 1))
        zkplus1_ztildeT = cp.Variable((m, m))
        zkplus1wT = cp.Variable((m, m))
        ztilde_ztildeT = cp.Variable((m, m))
        ztilde_wT = cp.Variable((m, m))

        constraints += [ztilde >= w, ztilde >= l,
                        cp.diag(ztilde_ztildeT - ztilde_wT - l @ ztilde.T + l @ w.T) == 0,
                        zkplus1 <= ztilde, zkplus1 <= u,
                        cp.diag(u @ ztilde.T - u @ zkplus1.T - zkplus1_ztildeT + zkplus1_zkplus1T) == 0,
                        cp.bmat([
                            [zkplus1_zkplus1T, zkplus1_ztildeT, zkplus1wT, zkplus1],
                            [zkplus1_ztildeT.T, ztilde_ztildeT, ztilde_wT, ztilde],
                            [zkplus1wT.T, ztilde_wT.T, wwT, w],
                            [zkplus1.T, ztilde.T, w.T, np.array([[1]])]
                        ]) >> 0]

        x_vars.append(xkplus1)
        xxT_vars.append(xkplus1_xkplus1T)
        y_vars.append(ykplus1)
        yyT_vars.append(ykplus1_ykplus1T)
        z_vars.append(zkplus1)
        zzT_vars.append(zkplus1_zkplus1T)

        xk = xkplus1
        xk_xkT = xkplus1_xkplus1T
        yk = ykplus1
        yk_ykT = ykplus1_ykplus1T
        zk = zkplus1
        zk_zkT = zkplus1_zkplus1T

        zk_ykT = ykplus1_zkplus1T.T
        zk_xkT = xkplus1_zkplus1T.T
        yk_xkT = xkplus1_ykplus1T.T

        xk_qT = xkplus1_qT

    xN = xk
    xN_xNT = xk_xkT
    xNminus1 = x_vars[-2]
    xNminus1_xNminus1T = xxT_vars[-2]
    # can reuse this one
    xN_xNminus1T = xkplus1_xkT

    zN = zk
    zN_zNT = zk_zkT
    zNminus1 = z_vars[-2]
    zNminus1_zNminus1T = zzT_vars[-2]

    zN_zNminus1T = cp.Variable((m, m))

    yN = yk
    yN_yNT = yk_ykT
    yNminus1 = y_vars[-2]
    yNminus1_yNminus1T = yyT_vars[-2]

    yN_yNminus1T = cp.Variable((m, m))

    constraints += [
        cp.bmat([
            [zN_zNT, zN_zNminus1T, zN],
            [zN_zNminus1T.T, zNminus1_zNminus1T, zNminus1],
            [zN.T, zNminus1.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [yN_yNT, yN_yNminus1T, yN],
            [yN_yNminus1T.T, yNminus1_yNminus1T, yNminus1],
            [yN.T, yNminus1.T, np.array([[1]])]
        ]) >> 0,
    ]

    # print(len(x_vars), len(xxT_vars), len(y_vars), len(yyT_vars), len(z_vars), len(zzT_vars))

    obj = cp.Maximize(cp.trace(xN_xNT) - 2 * cp.trace(xN_xNminus1T) + cp.trace(xNminus1_xNminus1T)
                      + cp.trace(yN_yNT) - 2 * cp.trace(yN_yNminus1T) + cp.trace(yNminus1_yNminus1T)
                      + cp.trace(zN_zNT) - 2 * cp.trace(zN_zNminus1T) + cp.trace(zNminus1_zNminus1T))
    # obj = cp.Maximize(cp.trace(P))
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=False)
    print(res)


def main():
    # test_Nstep_OSQP_SDR()
    # test_Nstep_OSQP_SDR_alt()
    test_Nstep_quad_boxproj()


if __name__ == '__main__':
    main()
