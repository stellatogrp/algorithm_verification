import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp
import time

from gurobipy import GRB


def eval_lasso(A, b, x, lambd):
    c = A @ x - b
    return .5 * c.T @ c + lambd * np.linalg.norm(x, 1)


def solve_LASSO_with_cvxpy(n, A, b, lambd):
    x = cp.Variable(n)
    obj = .5 * cp.sum_squares(A @ x - b) + lambd * cp.norm(x, 1)

    problem = cp.Problem(cp.Minimize(obj))
    result = problem.solve()

    print('-------- solving with cvxpy --------')
    print('result from cvxpy:', result)
    print('optimal x:', np.round(x.value, 4))
    print(eval_lasso(A, b, x.value, lambd))


def ISTA_PEP_onestep():
    np.random.seed(1)

    n = 9
    m = 16
    lambd = 5
    t = .05
    R = 1

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    I = spa.eye(n)
    minusI = -I
    ones = np.ones(n)
    lambd_ones = lambd * ones
    lambdt_ones = lambd * t * ones
    print(A)
    print(b)

    ATA = A.T @ A
    halfATA = .5 * ATA
    bTA = b.T @ A

    solve_LASSO_with_cvxpy(n, A, b, lambd)

    print('-------- solving with gurobi --------')
    m = gp.Model()
    m.setParam('NonConvex', 2)

    # v = m.addMVar(n,
    #               name='v',
    #               ub=gp.GRB.INFINITY * np.ones(n),
    #               lb=-gp.GRB.INFINITY * np.ones(n))

    x0 = m.addMVar(n,
                   name='x0',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))
    x1 = m.addMVar(n,
                   name='x1',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))

    # u, gamma1, gamm2 are all \ge 0, so the default [0, inf) constraints are fine
    u = m.addMVar(n, name='u')
    gamma1 = m.addMVar(n, name='gamma1')
    gamma2 = m.addMVar(n, name='gamma2')

    obj = x1 @ halfATA @ x1 - bTA @ x1 + lambd_ones @ u + .5 * b.T @ b
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(x0 @ I @ x0 <= R ** 2)

    # fake constraints
    m.addConstr(x1 @ I @ x1 <= 100 * R ** 2)
    m.addConstr(u @ I @ u <= 100 * R ** 2)

    m.addConstr(x1 - (I - t*ATA) @ x0 + gamma1 - gamma2 == t * A.T @ b)
    m.addConstr(gamma1 + gamma2 == lambdt_ones)
    # don't need gamma1, gamma2 >= 0 because defaults
    m.addConstr(x1 <= u)
    m.addConstr(-x1 <= u)
    m.addConstr(gamma1 @ x1 - gamma1 @ u == 0)
    m.addConstr(gamma2 @ x1 + gamma2 @ u == 0)

    start = time.time()
    m.optimize()
    total_time = time.time() - start

    print('x0:', np.round(x0.X, 4))
    print('x1:', np.round(x1.X, 4))
    print('u:', np.round(u.X, 4))
    print(eval_lasso(A, b, x1.X, lambd))
    return total_time


def ISTA_PEP_onestep_SOS_test():
    np.random.seed(1)

    n = 9
    m = 16
    lambd = 5
    t = .05
    R = 1

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    I = spa.eye(n)
    minusI = -I
    ones = np.ones(n)
    lambd_ones = lambd * ones
    lambdt_ones = lambd * t * ones
    print(A)
    print(b)

    ATA = A.T @ A
    halfATA = .5 * ATA
    bTA = b.T @ A

    solve_LASSO_with_cvxpy(n, A, b, lambd)

    print('-------- solving with gurobi --------')
    m = gp.Model()
    m.setParam('NonConvex', 2)

    # v = m.addMVar(n,
    #               name='v',
    #               ub=gp.GRB.INFINITY * np.ones(n),
    #               lb=-gp.GRB.INFINITY * np.ones(n))

    x0 = m.addMVar(n,
                   name='x0',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))
    x1 = m.addMVar(n,
                   name='x1',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))

    # u, gamma1, gamm2 are all \ge 0, so the default [0, inf) constraints are fine
    u = m.addMVar(n, name='u')
    gamma1 = m.addMVar(n, name='gamma1')
    gamma2 = m.addMVar(n, name='gamma2')

    z0 = m.addMVar(n,
                   name='z0',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))
    z1 = m.addMVar(n,
                   name='z1',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))

    obj = x1 @ halfATA @ x1 - bTA @ x1 + lambd_ones @ u + .5 * b.T @ b
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(x0 @ I @ x0 <= R ** 2)

    # fake constraints
    m.addConstr(x1 @ I @ x1 <= 100 * R ** 2)
    m.addConstr(u @ I @ u <= 100 * R ** 2)

    m.addConstr(x1 - (I - t*ATA) @ x0 + gamma1 - gamma2 == t * A.T @ b)
    m.addConstr(gamma1 + gamma2 == lambdt_ones)
    # don't need gamma1, gamma2 >= 0 because defaults

    # these are extra constraints to use the SOS constraints
    m.addConstr(z0 == x1 - u)
    m.addConstr(z1 == x1 + u)

    m.addConstr(x1 <= u)
    m.addConstr(-x1 <= u)
    for i in range(n):
        # m.addConstr(gamma1 @ x1 - gamma1 @ u == 0)
        m.addSOS(GRB.SOS_TYPE1, [gamma1[i], z0[i]], [1, 2])
        # m.addConstr(gamma2 @ x1 + gamma2 @ u == 0)
        m.addSOS(GRB.SOS_TYPE1, [gamma2[i], z1[i]], [1, 2])

    start = time.time()
    m.optimize()
    total_time = time.time() - start

    print('x0:', np.round(x0.X, 4))
    print('x1:', np.round(x1.X, 4))
    print('u:', np.round(u.X, 4))
    print(eval_lasso(A, b, x1.X, lambd))
    return total_time


def ISTA_PEP_N_steps(N):
    np.random.seed(0)

    n = 6
    m = 10
    lambd = 5
    t = .05
    R = 1
    max_bound_multiplier = 100

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    I = spa.eye(n)
    minusI = -I
    ones = np.ones(n)
    lambd_ones = lambd * ones
    lambdt_ones = lambd * t * ones

    ATA = A.T @ A
    halfATA = .5 * ATA
    bTA = b.T @ A

    solve_LASSO_with_cvxpy(n, A, b, lambd)

    print('-------- solving with gurobi for %d steps--------' % N)
    m = gp.Model()
    m.setParam('NonConvex', 2)

    x = m.addMVar((N+1, n),
                  name='x',
                  ub=gp.GRB.INFINITY * np.ones((N+1, n)),
                  lb=-gp.GRB.INFINITY * np.ones((N+1, n)))

    # pad the following with an extra col of 0's for readability
    u = m.addMVar((N+1, n), name='u')
    gamma1 = m.addMVar((N+1, n), name='gamma1')
    gamma2 = m.addMVar((N+1, n), name='gamma2')

    obj = x[N] @ halfATA @ x[N] - bTA @ x[N] + lambd_ones @ u[N] + .5 * b.T @ b
    m.setObjective(obj, GRB.MAXIMIZE)

    # Initial bound
    m.addConstr(x[0] @ I @ x[0] <= R ** 2)
    for i in range(1, N+1):
        # fake constraints
        m.addConstr(x[i] @ I @ x[i] <= max_bound_multiplier * R ** 2)
        m.addConstr(u[i] @ I @ u[i] <= max_bound_multiplier * R ** 2)

        m.addConstr(x[i] - (I - t * ATA) @ x[i-1] + gamma1[i] - gamma2[i] == t * A.T @ b)
        m.addConstr(gamma1[i] + gamma2[i] == lambdt_ones)
        # don't need gamma1, gamma2 >= 0 because defaults
        m.addConstr(x[i] <= u[i])
        m.addConstr(-x[i] <= u[i])
        m.addConstr(gamma1[i] @ x[i] - gamma1[i] @ u[i] == 0)
        m.addConstr(gamma2[i] @ x[i] + gamma2[i] @ u[i] == 0)

    m.optimize()


def main():
    t1 = ISTA_PEP_onestep()
    t2 = ISTA_PEP_onestep_SOS_test()
    print('time default:', t1)
    print('time with SOS:', t2)
    # N = 2
    # ISTA_PEP_N_steps(N)


if __name__ == '__main__':
    main()
