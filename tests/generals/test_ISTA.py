import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp

from gurobipy import GRB


def test_ISTA_SDR():
    N = 2
    m = 3
    n = 2
    R = 2
    lambd = 5
    eps = .1
    In = spa.eye(n)
    Zmn = spa.csc_matrix((m, n))
    Im = spa.eye(m)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A

    t = .05
    lambd_t_ones = lambd * t * np.ones(n).reshape(n, 1)
    x_vars = []
    xxT_vars = []

    x0 = cp.Variable((n, 1))
    x0x0T = cp.Variable((n, n))
    x_vars.append(x0)
    xxT_vars.append(x0x0T)

    b = cp.Variable((m, 1))
    bbT = cp.Variable((m, m))

    # inital constraints
    constraints = [
                    cp.sum_squares(x0) <= R ** 2, cp.trace(x0x0T) <= R ** 2,
                    # x0 == 0, x0x0T == 0,
                    cp.sum_squares(b) <= .5 * R ** 2, cp.trace(bbT) <= .5 * R ** 2,
                    # b == 0, bbT == 0,
                  ]

    C = spa.bmat([[In, t * ATA - In, -t * A.T]])
    H = spa.bmat([[In, In, -In]])

    xk = x0
    xk_xkT = x0x0T
    for i in range(N):
        ykplus1 = cp.Variable((n, 1))
        ykplus1_ykplus1T = cp.Variable((n, n))
        u = cp.vstack([ykplus1, xk, b])

        ykplus1_xkT = cp.Variable((n, n))
        ykplus1_bT = cp.Variable((n, m))

        xk_bT = cp.Variable((n, m))

        uuT = cp.bmat([
            [ykplus1_ykplus1T, ykplus1_xkT, ykplus1_bT],
            [ykplus1_xkT.T, xk_xkT, xk_bT],
            [ykplus1_bT.T, xk_bT.T, bbT]
        ])

        constraints += [C @ u == 0, C @ uuT @ C.T == 0,
                        cp.bmat([
                            [uuT, u],
                            [u.T, np.array([[1]])]
                        ]) >> 0]


        v = cp.Variable((n, 1))
        vvT = cp.Variable((n, n))
        w = cp.Variable((n, 1))
        wwT = cp.Variable((n, n))


        constraints += [v >= 0, v >= ykplus1 - lambd_t_ones, vvT >= 0,
                        w >= 0, w >= -ykplus1 - lambd_t_ones, wwT >= 0]
        v_ykplus1T = cp.Variable((n, n))
        w_ykplus1T = cp.Variable((n, n))
        wvT = cp.Variable((n, n))
        constraints += [cp.diag(vvT) == cp.diag(v_ykplus1T) - cp.diag(v @ lambd_t_ones.T),
                        cp.diag(wwT) == -cp.diag(w_ykplus1T) - cp.diag(w @ lambd_t_ones.T),
                        cp.bmat([
                            [wwT, wvT, w_ykplus1T, w],
                            [wvT.T, vvT, v_ykplus1T, v],
                            [w_ykplus1T.T, v_ykplus1T.T, ykplus1_ykplus1T, ykplus1],
                            [w.T, v.T, ykplus1.T, np.array([[1]])]
                        ]) >> 0]

        xkplus1 = cp.Variable((n, 1))
        xkplus1_xkplus1T = cp.Variable((n, n))
        xkplus1_w = cp.Variable((n, n))
        xkplus1_v = cp.Variable((n, n))

        p = cp.vstack([xkplus1, w, v])
        P = cp.bmat([
            [xkplus1_xkplus1T, xkplus1_w, xkplus1_v],
            [xkplus1_w.T, wwT, wvT],
            [xkplus1_v.T, wvT.T, vvT]
        ])

        constraints += [H @ p == 0, H @ P @ H.T == 0,
                           cp.bmat([
                               [P, p],
                               [p.T, np.array([[1]])]
                           ]) >> 0,
                        ]
        # xkplus1 = v - w
        # xkplus1_xkplus1T = vvT - wvT.T - wvT + wwT

        x_vars.append(xkplus1)
        xxT_vars.append(xkplus1_xkplus1T)

        xk = xkplus1
        xk_xkT = xkplus1_xkplus1T

    xN = xk
    xN_xNT = xk_xkT
    xNminus1 = x_vars[-2]
    xNminus1_xNminus1T = xxT_vars[-2]

    xN_xNminus1T = cp.Variable((n, n))

    constraints += [cp.bmat([
        [xN_xNT, xN_xNminus1T, xN],
        [xN_xNminus1T.T, xNminus1_xNminus1T, xNminus1],
        [xN.T, xNminus1.T, np.array([[1]])]
    ]) >> 0]

    obj = cp.Maximize(cp.trace(xN_xNT) - 2 * cp.trace(xN_xNminus1T) + cp.trace(xNminus1_xNminus1T))
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=True)
    print(res)


def test_ISTA_Gurobi():
    pass


def test_Nstep_ISTA_state_SDR():
    N = 1
    m = 3
    n = 2
    R = 2
    eps = .1
    lambd = 5
    t = .05
    In = spa.eye(n)
    Zn = spa.csc_matrix((n, n))
    Zm = spa.csc_matrix((m, m))
    Znm = spa.csc_matrix((n, m))
    Zmn = spa.csc_matrix((m, n))
    Im = spa.eye(m)
    lambd_t_ones = lambd * t * np.ones(n).reshape(n, 1)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A

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

    x0y0T = cp.Variable((n, n))
    x0v0T = cp.Variable((n, n))
    x0w0T = cp.Variable((n, n))
    x0bT = cp.Variable((n, m))

    y0v0T = cp.Variable((n, n))
    y0w0T = cp.Variable((n, n))
    y0bT = cp.Variable((n, m))

    v0w0T = cp.Variable((n, n))
    v0bT = cp.Variable((n, m))

    w0bT = cp.Variable((n, m))

    s0 = cp.vstack([x0, y0, v0, w0])
    s0s0T = cp.bmat([
        [x0x0T, x0y0T, x0v0T, x0w0T],
        [x0y0T.T, y0y0T, y0v0T, y0w0T],
        [x0v0T.T, y0v0T.T, v0v0T, v0w0T],
        [x0w0T.T, y0w0T.T, v0w0T.T, w0w0T],
    ])

    C = spa.bmat([[Zn, In, Zn, Zn, t * ATA - In, Zn, Zn, Zn, -t * A.T]])
    H = spa.bmat([[In, Zn, -In, In, Zn, Zn, Zn, Zn, Znm]])
    print(C.shape, H.shape)

    constraints = [
        cp.sum_squares(x0) <= R ** 2, cp.trace(x0x0T) <= R ** 2,
        cp.sum_squares(b) <= .5 * R ** 2, cp.trace(bbT) <= .5 * R ** 2,
        v0 == 0, v0v0T == 0,
        w0 == 0, w0w0T == 0,
    ]

    s_vars = [s0]
    ssT_vars = [s0s0T]
    scross_vars = [0]

    state_n = 4 * n
    sk = s0
    sk_skT = s0s0T
    sk_bT = cp.vstack([x0bT, y0bT, v0bT, w0bT])
    for k in range(N):
        xkplus1 = cp.Variable((n, 1))
        xkplus1_xkplus1T = cp.Variable((n, n))

        ykplus1 = cp.Variable((n, 1))
        ykplus1_ykplus1T = cp.Variable((n, n))

        vkplus1 = cp.Variable((n, 1))
        vkplus1_vkplus1T = cp.Variable((n, n))

        wkplus1 = cp.Variable((n, 1))
        wkplus1_wkplus1T = cp.Variable((n, n))

        xkplus1_ykplus1T = cp.Variable((n, n))
        xkplus1_vkplus1T = cp.Variable((n, n))
        xkplus1_wkplus1T = cp.Variable((n, n))
        xkplus1_bT = cp.Variable((n, m))

        ykplus1_vkplus1T = cp.Variable((n, n))
        ykplus1_wkplus1T = cp.Variable((n, n))
        ykplus1_bT = cp.Variable((n, m))

        vkplus1_wkplus1T = cp.Variable((n, n))
        vkplus1_bT = cp.Variable((n, m))

        wkplus1_bT = cp.Variable((n, m))

        skplus1 = cp.vstack([xkplus1, ykplus1, vkplus1, wkplus1])
        skplus1_skplus1T = cp.bmat([
            [xkplus1_xkplus1T, xkplus1_ykplus1T, xkplus1_vkplus1T, xkplus1_wkplus1T],
            [xkplus1_ykplus1T.T, ykplus1_ykplus1T, ykplus1_vkplus1T, ykplus1_wkplus1T],
            [xkplus1_vkplus1T.T, ykplus1_vkplus1T.T, vkplus1_vkplus1T, vkplus1_wkplus1T],
            [xkplus1_wkplus1T.T, ykplus1_wkplus1T.T, vkplus1_wkplus1T.T, wkplus1_wkplus1T]
        ])
        skplus1_bT = cp.vstack([xkplus1_bT, ykplus1_bT, vkplus1_bT, wkplus1_bT])
        skplus1_skT = cp.Variable((state_n, state_n))

        u = cp.vstack([skplus1, sk, b])
        uuT = cp.bmat([
            [skplus1_skplus1T, skplus1_skT, skplus1_bT],
            [skplus1_skT.T, sk_skT, sk_bT],
            [skplus1_bT.T, sk_bT.T, bbT]
        ])

        # put in constraints
        constraints += [
            cp.bmat([
                [uuT, u],
                [u.T, np.array([[1]])]
            ]) >> 0,
            C @ u == 0, C @ uuT @ C.T == 0,
        ]

        constraints += [
            vkplus1 >= 0, vkplus1 >= ykplus1 - lambd_t_ones, vkplus1_vkplus1T >= 0,
            cp.diag(vkplus1_vkplus1T) == cp.diag(ykplus1_vkplus1T.T - vkplus1 @ lambd_t_ones.T),

            wkplus1 >= 0, wkplus1 >= -ykplus1, wkplus1_wkplus1T >= 0,
            cp.diag(wkplus1_wkplus1T) == cp.diag(-ykplus1_wkplus1T.T - wkplus1 @ lambd_t_ones.T),

            H @ u == 0, H @ uuT @ H.T == 0,
        ]
        #

        s_vars.append(skplus1)
        ssT_vars.append(skplus1_skplus1T)
        scross_vars.append(skplus1_skT)

        sk = skplus1
        sk_skT = skplus1_skplus1T
        sk_bT = skplus1_bT

    sN = ssT_vars[N]
    sNminus1 = ssT_vars[N-1]
    scross = scross_vars[N]

    # print(sN.shape, sNminus1.shape, scross.shape)

    xN_xNT = sN[0: n, 0: n]
    xNminus1_xNminus1T = sNminus1[0: n, 0: n]
    xN_xNminus1T = scross[0: n, 0: n]

    obj = cp.Maximize(cp.trace(xN_xNT - 2 * xN_xNminus1T + xNminus1_xNminus1T))
    problem = cp.Problem(obj, constraints)
    res = problem.solve(solver=cp.MOSEK, verbose=True)
    print(res)


def main():
    # test_ISTA_SDR()
    test_Nstep_ISTA_state_SDR()


if __name__ == '__main__':
    main()
