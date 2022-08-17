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


def test_2step_NNLS_splitting(N=1, m=5, n=3, R=1):
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
    C = spa.bmat([[In - t * ATA, t * A.T]])


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
