import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp

from gurobipy import GRB


def test_Nstep_NNLS_Gurobi(N=1, m=5, n=3, R=1):
    print('solving nonconvex QCQP with N = %d' % N)

    eps = .1
    In = spa.eye(n)
    twoIn = 2 * In
    Zmn = spa.csc_matrix((m, n))
    Im = spa.eye(m)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A

    t = .05

    # print('-------- solving with gurobi alternate--------')
    model = gp.Model()
    model.setParam('NonConvex', 2)
    # model.setParam('MIPFocus', 3)

    x_lb = np.zeros((N + 1, n))
    x_lb[0] = -gp.GRB.INFINITY * np.ones(n)
    print(x_lb)
    # exit(0)
    x = model.addMVar((N + 1, n),
                   name='x',
                   ub=gp.GRB.INFINITY * np.ones((N + 1, n)),
                   lb=-gp.GRB.INFINITY * np.ones((N + 1, n)))
                   # lb=x_lb)

    y = model.addMVar((N + 1, n),
                  name='y',
                  ub=gp.GRB.INFINITY * np.ones((N + 1, n)),
                  lb=-gp.GRB.INFINITY * np.ones((N + 1, n)))

    z = model.addMVar((N + 1, n),
                  name='z',
                  ub=gp.GRB.INFINITY * np.ones((N + 1, n)),
                  lb=-gp.GRB.INFINITY * np.ones((N + 1, n)))

    b = model.addMVar(m,
                  name='b',
                  # ub=3 * np.ones(m),
                  # lb=1 * np.ones(m))
                  ub=gp.GRB.INFINITY * np.ones(m),
                  lb=-gp.GRB.INFINITY * np.ones(m))

    model.addConstr(x[0] @ In @ x[0] <= R ** 2)
    model.addConstr(b @ Im @ b <= R ** 2)
    # model.addConstr(1 <= b)
    # model.addConstr(b <= 3)
    # model.addConstr(x[0] == 0)
    # model.addConstr(b @ Im @ b <= (.1 * R ** 2))
    for k in range(N):
        # print(x[k].shape)
        model.addConstr(y[k+1] == (In - t * ATA) @ x[k] + t * A.T @ b)
        model.addConstr(x[k+1] >= 0)
        model.addConstr(x[k+1] >= y[k+1])

        model.addConstr(z[k+1] == x[k+1] - y[k+1])
        model.addConstr(x[k+1] @ z[k+1] == 0)
        # model.addConstr(x[k+1] @ x[k+1] - x[k+1] @ y[k+1] == 0)
        # for i in range(n):
        #     model.addConstr(x[k+1, i] * x[k+1, i] == x[k+1, i] * y[k+1, i])

    obj = x[N] @ In @ x[N] - x[N] @ twoIn @ x[N-1] + x[N-1] @ In @ x[N-1]
    model.setObjective(obj, GRB.MAXIMIZE)

    model.optimize()
    print('y = ', y.X)
    print('x = ', x.X)
    # print('gap =', model.MIPGap)


def sample_solve_via_cvxpy():
    np.random.seed(0)
    m = 10
    n = 5
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A

    # R = 50
    #
    # r = np.random.randn(m)
    # r = r / np.linalg.norm(r)
    # Rvec = np.random.uniform(low=0, high=.1 * R ** 2)
    # print(Rvec, r)
    #
    # b = Rvec * r
    # print(np.linalg.norm(b))

    b = np.random.uniform(1, 3, m)

    x = cp.Variable(n)
    obj = cp.Minimize(cp.sum_squares(A @ x - b))
    constraints = [x >= 0]
    prob = cp.Problem(obj, constraints)
    res = prob.solve()
    print(x.value)


def test_Nstep_NNLS_SDR(N=1, m=5, n=3, R=1):
    print('solving SDR with N = %d' % N)
    eps = .2
    In = spa.eye(n)
    Zmn = spa.csc_matrix((m, n))
    Im = spa.eye(m)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A


    t = .05
    x_vars = []
    xxT_vars = []
    xcross_vars = []
    obj_vars = []

    x0 = cp.Variable((n, 1))
    x0x0T = cp.Variable((n, n))
    x_vars.append(x0)
    xxT_vars.append(x0x0T)

    b = cp.Variable((m, 1))
    bbT = cp.Variable((m, m))

    l = 1
    u = 3
    # x_val = np.array([0, .3884, 0, .0794, .3512]).reshape(n, 1)

    # inital constraints
    constraints = [
                    cp.sum_squares(x0) <= R ** 2, cp.trace(x0x0T) <= R ** 2,
                    # x0 == x_val, x0x0T == np.outer(x_val, x_val),
                    # cp.trace(x0x0T) - 2 * x_val.T @ x0 + x_val.T @ x_val <= eps ** 2,
                    # cp.sum_squares(b) <= .1 * R ** 2, cp.trace(bbT) <= .1 * R ** 2,
                    # cp.diag(P[n: 2 * n, n: 2 * n]) <= (l + u) * P[n: 2 * n, -1] - l * u
                    cp.reshape(cp.diag(bbT), (m, 1)) <= (l + u) * b - l * u,
                    # b == 0, bbT == 0,
                  ]

    C = spa.bmat([[In, t * ATA - In, -t * A.T]])

    xk = x0
    xk_xkT = x0x0T
    for i in range(N):
        ykplus1 = cp.Variable((n, 1))
        ykplus1_ykplus1T = cp.Variable((n, n))
        u = cp.vstack([ykplus1, xk, b])
        block_n = u.shape[0]

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

        xkplus1 = cp.Variable((n, 1))
        xkplus1_xkplus1T = cp.Variable((n, n))

        constraints += [xkplus1 >= 0, xkplus1 >= ykplus1, xkplus1_xkplus1T >= 0]
        xkplus1_ykplus1T = cp.Variable((n, n))
        constraints += [cp.diag(xkplus1_xkplus1T) == cp.diag(xkplus1_ykplus1T),
                        cp.bmat([
                            [xkplus1_xkplus1T, xkplus1_ykplus1T, xkplus1],
                            [xkplus1_ykplus1T.T, ykplus1_ykplus1T, ykplus1],
                            [xkplus1.T, ykplus1.T, np.array([[1]])]
                        ]) >> 0]

        xkplus1_xkT = cp.Variable((n, n))
        constraints += [
            cp.bmat([
                [xkplus1_xkplus1T, xkplus1_xkT, xkplus1],
                [xkplus1_xkT.T, xk_xkT, xk],
                [xkplus1.T, xk.T, np.array([[1]])]
            ]) >> 0,
        ]
        obj_vars.append(cp.trace(xkplus1_xkplus1T - 2 * xkplus1_xkT + xk_xkT))

        x_vars.append(xkplus1)
        xxT_vars.append(xkplus1_xkplus1T)
        xcross_vars.append(xkplus1_xkT)

        xk = xkplus1
        xk_xkT = xkplus1_xkplus1T

    # xN = x_vars[-1]
    # xN_xNT = xxT_vars[-1]
    # xNminus1 = x_vars[-2]
    # xNminus1_xNminus1T = xxT_vars[-2]
    #
    # # xN_xNminus1T = cp.Variable((n, n))
    # xN_xNminus1T = xcross_vars[-1]
    #
    # constraints += [cp.bmat([
    #     [xN_xNT, xN_xNminus1T, xN],
    #     [xN_xNminus1T.T, xNminus1_xNminus1T, xNminus1],
    #     [xN.T, xNminus1.T, np.array([[1]])]
    # ]) >> 0]

    if N >= 2:
        for k in range(1, N):
            print(obj_vars[k])
            print(obj_vars[k-1])
            constraints.append(obj_vars[k] <= obj_vars[k-1])


    # obj = cp.Maximize(cp.trace(xN_xNT) - 2 * cp.trace(xN_xNminus1T) + cp.trace(xNminus1_xNminus1T))
    obj = cp.Maximize(obj_vars[-1])
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=True)
    print('res =', res)
    print('comp time =', prob.solver_stats.solve_time)
    print(np.round(x0.value, 4))


def test_1step_NNLS_no_y(m=5, n=3, R=1):
    In = spa.eye(n)
    Zmn = spa.csc_matrix((m, n))
    Im = spa.eye(m)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A

    t = .05

    x0 = cp.Variable((n, 1))
    x0x0T = cp.Variable((n, n))
    x1 = cp.Variable((n, 1))
    x1x1T = cp.Variable((n, n))
    x1x0T = cp.Variable((n, n))
    x1bT = cp.Variable((n, m))

    b = cp.Variable((m, 1))
    bbT = cp.Variable((m, m))

    l = 1
    u = 3

    # inital constraints
    constraints = [
        cp.sum_squares(x0) <= R ** 2, cp.trace(x0x0T) <= R ** 2,
        # cp.reshape(cp.diag(bbT), (m, 1)) <= (l + u) * b - l * u,
        cp.sum_squares(b) <= R ** 2, cp.trace(bbT) <= R ** 2,
    ]

    constraints += [
        x1 >= 0, x1x1T >= 0,
        x1 >= (In - t * ATA) @ x0 + t * A.T @ b,
        cp.diag(x1x1T) == cp.diag(x1x0T @ (In - t * ATA).T + t * x1bT @ A)
    ]

    constraints += [
        cp.bmat([
            [x1x1T, x1x0T, x1],
            [x1x0T.T, x0x0T, x0],
            [x1.T, x0.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [x1x1T, x1bT, x1],
            [x1bT.T, bbT, b],
            [x1.T, b.T, np.array([[1]])]
        ]) >> 0
    ]

    obj = cp.Maximize(cp.trace(x1x1T - 2 * x1x0T + x0x0T))
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=True)
    print('res =', res)
    print('comp time =', prob.solver_stats.solve_time)
    print(np.round(x0.value, 4))


def test_2step_NNLS_no_y(m=5, n=3, R=1):
    In = spa.eye(n)
    Zmn = spa.csc_matrix((m, n))
    Im = spa.eye(m)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A

    t = .05

    x0 = cp.Variable((n, 1))
    x0x0T = cp.Variable((n, n))
    x1 = cp.Variable((n, 1))
    x1x1T = cp.Variable((n, n))
    x1x0T = cp.Variable((n, n))
    x1bT = cp.Variable((n, m))

    x2 = cp.Variable((n, 1))
    x2x2T = cp.Variable((n, n))
    x2x1T = cp.Variable((n, n))
    x2bT = cp.Variable((n, m))

    b = cp.Variable((m, 1))
    bbT = cp.Variable((m, m))

    l = 1
    u = 3

    # inital constraints
    constraints = [
        cp.sum_squares(x0) <= R ** 2, cp.trace(x0x0T) <= R ** 2,
        cp.reshape(cp.diag(bbT), (m, 1)) <= (l + u) * b - l * u,
    ]

    constraints += [
        x1 >= 0, x1x1T >= 0,
        x1 >= (In - t * ATA) @ x0 + t * A.T @ b,
        cp.diag(x1x1T) == cp.diag(x1x0T @ (In - t * ATA).T + t * x1bT @ A),
        x2 >= 0, x2x2T >= 0,
        x2 >= (In - t * ATA) @ x1 + t * A.T @ b,
        cp.diag(x2x2T) == cp.diag(x2x1T @ (In - t * ATA).T + t * x2bT @ A)
    ]

    constraints += [
        cp.bmat([
            [x1x1T, x1x0T, x1],
            [x1x0T.T, x0x0T, x0],
            [x1.T, x0.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [x1x1T, x1bT, x1],
            [x1bT.T, bbT, b],
            [x1.T, b.T, np.array([[1]])]
        ]) >> 0,

        cp.bmat([
            [x2x2T, x2x1T, x2],
            [x2x1T.T, x1x1T, x1],
            [x2.T, x1.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [x2x2T, x2bT, x2],
            [x2bT.T, bbT, b],
            [x2.T, b.T, np.array([[1]])]
        ]) >> 0,
    ]

    obj = cp.Maximize(cp.trace(x2x2T - 2 * x2x1T + x1x1T))
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=True)
    print('res =', res)
    print('comp time =', prob.solver_stats.solve_time)
    print(np.round(x0.value, 4))


def test_3step_NNLS_no_y(m=5, n=3, R=1):
    In = spa.eye(n)
    Zmn = spa.csc_matrix((m, n))
    Im = spa.eye(m)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A

    t = .05

    x0 = cp.Variable((n, 1))
    x0x0T = cp.Variable((n, n))
    x1 = cp.Variable((n, 1))
    x1x1T = cp.Variable((n, n))
    x1x0T = cp.Variable((n, n))
    x1bT = cp.Variable((n, m))

    x2 = cp.Variable((n, 1))
    x2x2T = cp.Variable((n, n))
    x2x1T = cp.Variable((n, n))
    x2bT = cp.Variable((n, m))

    x3 = cp.Variable((n, 1))
    x3x3T = cp.Variable((n, n))
    x3x2T = cp.Variable((n, n))
    x3bT = cp.Variable((n, m))

    b = cp.Variable((m, 1))
    bbT = cp.Variable((m, m))

    l = 1
    u = 3

    # inital constraints
    constraints = [
        cp.sum_squares(x0) <= R ** 2, cp.trace(x0x0T) <= R ** 2,
        cp.reshape(cp.diag(bbT), (m, 1)) <= (l + u) * b - l * u,
    ]

    constraints += [
        x1 >= 0, x1x1T >= 0,
        x1 >= (In - t * ATA) @ x0 + t * A.T @ b,
        cp.diag(x1x1T) == cp.diag(x1x0T @ (In - t * ATA).T + t * x1bT @ A),
        x2 >= 0, x2x2T >= 0,
        x2 >= (In - t * ATA) @ x1 + t * A.T @ b,
        cp.diag(x2x2T) == cp.diag(x2x1T @ (In - t * ATA).T + t * x2bT @ A),
        x3 >= 0, x3x3T >= 0,
        x3 >= (In - t * ATA) @ x2 + t * A.T @ b,
        cp.diag(x3x3T) == cp.diag(x3x2T @ (In - t * ATA).T + t * x3bT @ A),
    ]

    constraints += [
        cp.bmat([
            [x1x1T, x1x0T, x1],
            [x1x0T.T, x0x0T, x0],
            [x1.T, x0.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [x1x1T, x1bT, x1],
            [x1bT.T, bbT, b],
            [x1.T, b.T, np.array([[1]])]
        ]) >> 0,

        cp.bmat([
            [x2x2T, x2x1T, x2],
            [x2x1T.T, x1x1T, x1],
            [x2.T, x1.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [x2x2T, x2bT, x2],
            [x2bT.T, bbT, b],
            [x2.T, b.T, np.array([[1]])]
        ]) >> 0,

        cp.bmat([
            [x3x3T, x3x2T, x3],
            [x3x2T.T, x2x2T, x2],
            [x3.T, x2.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [x3x3T, x3bT, x3],
            [x3bT.T, bbT, b],
            [x3.T, b.T, np.array([[1]])]
        ]) >> 0,
    ]

    obj = cp.Maximize(cp.trace(x3x3T - 2 * x3x2T + x3x3T))
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=True)
    print('res =', res)
    print('comp time =', prob.solver_stats.solve_time)
    print(np.round(x0.value, 4))


def test_1step_NNLS_linking_y(m=5, n=3, R=1):
    print('double l2 bounds')
    In = spa.eye(n)
    Zmn = spa.csc_matrix((m, n))
    Im = spa.eye(m)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A

    print(A)
    t = .05

    x0 = cp.Variable((n, 1))
    x0x0T = cp.Variable((n, n))
    x0bT = cp.Variable((n, m))

    x1 = cp.Variable((n, 1))
    x1x1T = cp.Variable((n, n))
    x1x0T = cp.Variable((n, n))
    x1bT = cp.Variable((n, m))

    # y1 = cp.Variable((n, 1))
    # y1y1T = cp.Variable((n, n))
    # x1y1T = cp.Variable((n, n))
    y1x0T = cp.Variable((n, n))
    y1bT = cp.Variable((n, m))

    b = cp.Variable((m, 1))
    bbT = cp.Variable((m, m))

    l = 1
    u = 3

    constraints = [
        cp.sum_squares(x0) <= R ** 2, cp.trace(x0x0T) <= R ** 2,
        cp.reshape(cp.diag(bbT), (m, 1)) <= (l + u) * b - l * u,
        # cp.sum_squares(b) <= R ** 2, cp.trace(bbT) <= R ** 2,
    ]

    C = spa.bmat([[In - t * ATA, t * A.T]])

    u1 = cp.vstack([x0, b])
    u1u1T = cp.bmat([
        [x0x0T, x0bT],
        [x0bT.T, bbT]
    ])
    x1u1T = cp.bmat([
        [x1x0T, x1bT]
    ])
    # x1u1T = cp.Variable((n, n + m))  # this is wrong, just for comparison sake

    y1 = C @ u1
    y1y1T = C @ u1u1T @ C.T
    y1u1T = C @ u1u1T
    x1y1T = x1u1T @ C.T
    constraints += [y1u1T == cp.bmat([
            [y1x0T, y1bT]
        ])
    ]
    # x1y1T = cp.Variable((n, n))  # this is wrong, just for comparison sake

    constraints += [
        x1 >= 0, x1x1T >= 0,
        x1 >= y1,
        cp.diag(x1x1T - x1y1T) == 0,
        # cp.diag(x1x1T) == cp.diag(x1x0T @ (In - t * ATA).T + t * x1bT @ A),
    ]

    # constraints += [
    #     cp.bmat([
    #         [x1x1T, x1y1T, x1u1T, x1],
    #         [x1y1T.T, y1y1T, y1u1T, y1],
    #         [x1u1T.T, y1u1T.T, u1u1T, u1],
    #         [x1.T, y1.T, u1.T, np.array([[1]])]
    #     ]) >> 0
    # ]

    y1x1T = x1y1T.T

    constraints += [
        cp.bmat([
            [x1x1T, x1u1T, x1],
            [x1u1T.T, u1u1T, u1],
            [x1.T, u1.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [y1y1T, y1u1T, y1],
            [y1u1T.T, u1u1T, u1],
            [y1.T, u1.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [x1x1T, x1y1T, x1],
            [y1x1T, y1y1T, y1],
            [x1.T, y1.T, np.array([[1]])]
        ]) >> 0,
    ]

    obj = cp.Maximize(cp.trace(x1x1T - 2 * x1x0T + x0x0T))
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=True)
    print('res =', res)
    print('comp time =', prob.solver_stats.solve_time)
    # print('x0 =', np.round(x0.value, 4))
    # print('b =', np.round(b.value, 4))


def main():
    N = 1
    m = 5
    n = 3
    R = 1
    test_Nstep_NNLS_Gurobi(N=N, m=m, n=n, R=R)
    # test_Nstep_NNLS_SDR(N=N, m=m, n=n, R=R)
    # test_1step_NNLS_no_y(m=m, n=n, R=R)
    # test_1step_NNLS_linking_y(m=m, n=n, R=R)
    # test_2step_NNLS_no_y(m=m, n=n, R=R)
    # test_3step_NNLS_no_y(m=m, n=n, R=R)

    # for N in range(10):
    #     print(N+1)
    #     test_Nstep_NNLS_SDR(N=N+1)
    # sample_solve_via_cvxpy()


if __name__ == '__main__':
    main()
