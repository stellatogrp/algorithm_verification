import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp

from gurobipy import GRB


def solve_NNLS_with_cvxpy(n, A, b):
    x = cp.Variable(n)
    obj = .5 * cp.sum_squares(A @ x - b)
    constraints = [x >= 0]

    problem = cp.Problem(cp.Minimize(obj), constraints)
    result = problem.solve()

    print('-------- solving with cvxpy --------')
    print('result from cvxpy:', result)
    print('optimal x:', np.round(x.value, 4))


def NNLS_PEP_onestep():
    np.random.seed(0)

    n = 2
    m = 3
    t = .05
    R = 1

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    I = spa.eye(n)

    print(A, b)

    solve_NNLS_with_cvxpy(n, A, b)

    ATA = A.T @ A
    halfATA = .5 * ATA
    bTA = b.T @ A

    print('-------- solving with gurobi --------')
    m = gp.Model()
    m.setParam('NonConvex', 2)

    x0 = m.addMVar(n,
                   name='x0',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))

    # x1, gamma are \ge 0 so default [0, inf) constraints are fine
    x1 = m.addMVar(n, name='u')
    gamma = m.addMVar(n, name='gamma1')

    obj = x1 @ halfATA @ x1 - bTA @ x1 + .5 * b.T @ b
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(x0 @ I @ x0 <= R ** 2)

    m.addConstr(x1 - (I - t * ATA) @ x0 - gamma == t * A.T @ b)
    m.addConstr(gamma @ x1 == 0)

    m.optimize()

    print('x0:', np.round(x0.X, 4))
    print('x1:', np.round(x1.X, 4))

    print(.5 * np.linalg.norm(A @ x1.X - b) ** 2)


def NNLS_PEP_onestep_alt():
    np.random.seed(0)

    n = 2
    m = 3
    t = .05
    R = 1

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    I = spa.eye(n)

    print(A, b)

    solve_NNLS_with_cvxpy(n, A, b)

    ATA = A.T @ A
    halfATA = .5 * ATA
    bTA = b.T @ A

    print('-------- solving with gurobi alternate--------')
    m = gp.Model()
    m.setParam('NonConvex', 2)

    x0 = m.addMVar(n,
                   name='x0',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))
    x1 = m.addMVar(n,
                   name='x1') # [0, inf) by default
    y1 = m.addMVar(n,
                   name='y1',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))

    obj = x1 @ halfATA @ x1 - bTA @ x1 + .5 * b.T @ b
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(x0 @ I @ x0 <= R ** 2)
    m.addConstr(y1 == (I - t * ATA) @ x0 + t * A.T @ b)
    m.addConstr(x1 >= y1)
    for i in range(n):
        m.addConstr(x1[i] * x1[i] == x1[i] * y1[i])

    m.optimize()
    print('x0:', np.round(x0.X, 4))
    print('x1:', np.round(x1.X, 4))
    print('y1:', np.round(y1.X, 4))


def main():
    # NNLS_PEP_onestep()
    NNLS_PEP_onestep_alt()


if __name__ == '__main__':
    main()
