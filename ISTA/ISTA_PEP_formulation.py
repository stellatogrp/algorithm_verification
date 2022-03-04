import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp

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
    np.random.seed(0)

    n = 5
    m = 10
    lambd = 1
    t = .05
    R = 1

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

    solve_LASSO_with_cvxpy(n, A, b, 1)

    print('-------- solving with gurobi --------')
    m = gp.Model()
    m.setParam('NonConvex', 2)

    # v = m.addMVar(n,
    #               name='v',
    #               ub=gp.GRB.INFINITY * np.ones(n),
    #               lb=-gp.GRB.INFINITY * np.ones(n))

    x0 = m.addMVar(n,
                   name='x0',
                   ub=(R ** 2) * np.ones(n),
                   lb=-(R ** 2) * np.ones(n))
    x1 = m.addMVar(n,
                   name='x1',
                   ub=(R ** 2) * np.ones(n),
                   lb=-(R ** 2) * np.ones(n))

    # u, gamma1, gamm2 are all \ge 0, so the default [0, inf) constraints are fine
    u = m.addMVar(n, name='u')
    gamma1 = m.addMVar(n, name='gamma1')
    gamma2 = m.addMVar(n, name='gamma2')

    obj = x1 @ halfATA @ x1 - bTA @ x1 + lambd_ones @ u + .5 * b.T @ b
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(x0 @ I @ x0 <= R ** 2)
    m.addConstr(x1 - (I - t*ATA) @ x0 + gamma1 - gamma2 == t * A.T @ b)
    m.addConstr(gamma1 + gamma2 == lambdt_ones)
    # don't need gamma1, gamma2 >= 0 because defaults
    m.addConstr(x1 <= u)
    m.addConstr(-x1 <= u)
    m.addConstr(gamma1 @ x1 - gamma1 @ u == 0)
    m.addConstr(gamma2 @ x1 + gamma2 @ u == 0)

    m.optimize()

    print('x0:', np.round(x0.X, 4))
    print('x1:', np.round(x1.X, 4))
    print(eval_lasso(A, b, x1.X, lambd))


def main():
    ISTA_PEP_onestep()


if __name__ == '__main__':
    main()
