import numpy as np
import cvxpy as cp
#  import scipy.sparse#   as spa
import gurobipy as gp
#  from qcqp2quad_form.quad_extractor import QuadExtractor
#  from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
#  from quadratic_functions.extended_slemma_sdp import *
#
#  from gurobipy import GRB


def test_box_proj_gurobi():
    n = 6
    np.random.seed(0)
    x = np.random.randn(n)
    print(x)
    l = -.5
    u = 1

    m = gp.Model()
    m.setParam('NonConvex', 2)

    y = m.addMVar(n,
                  name='y',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))
    z = m.addMVar(n,
                  name='z',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))
    t1 = m.addMVar(n,
                  name='t1',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))
    t2 = m.addMVar(n,
                   name='t2',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))

    t3 = m.addMVar(n,
                   name='t3',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))
    t4 = m.addMVar(n,
                   name='t4',
                   ub=gp.GRB.INFINITY * np.ones(n),
                   lb=-gp.GRB.INFINITY * np.ones(n))

    obj = 0
    m.setObjective(obj)
    m.addConstr(y >= l)
    m.addConstr(y >= x)
    m.addConstr(t1 == y - l)
    m.addConstr(t2 == y - x)
    m.addConstr(t1 @ t2 == 0)

    m.addConstr(z <= u)
    m.addConstr(z <= y)
    m.addConstr(t3 == u - z)
    m.addConstr(t4 == y - z)
    m.addConstr(t3 @ t4 == 0)

    m.optimize()
    print("")
    print('original x:', np.round(x, 4))
    print('y:', np.round(y.X, 4))
    print('z:', np.round(z.X, 4))


def test_relu_SDR_blocks():
    print('--------testing relu with SDR blocks--------')
    n = 2
    R = 1
    # x = np.array([-1, 1])
    # xxT = np.outer(x, x)
    A = np.random.randn(n, n)
    A = (A + A.T) / 2
    print("A = \n", A)

    x = cp.Variable((n, 1))  # Define as 2d array to simplify bmat
    xxT = cp.Variable((n, n), symmetric=True)

    z = cp.Variable((n, 1))  # Define as 2d array to simplify bmat
    zzT = cp.Variable((n, n), symmetric=True)
    zxT = cp.Variable((n, n))

    obj = cp.Maximize(cp.trace(A @ zzT))
    P = cp.Variable((2 * n + 1, 2 * n + 1), symmetric=True)
    constraints = [z >= x, z >= 0, cp.diag(zzT) == cp.diag(zxT), P >> 0,
                   cp.sum_squares(x) <= R ** 2, cp.trace(xxT) <= R ** 2]

    constraints += [P == cp.bmat([[zzT, zxT, z],
                                  [zxT.T, xxT, x],
                                  [z.T, x.T, np.array([[1]])]])]

    prob = cp.Problem(obj, constraints)
    res = prob.solve()
    print("objective = ", res)
    print("P = \n", np.round(P.value, 4))


def test_box_proj_SDR():
    print('--------testing SDR--------')
    n = 2
    R = 3
    np.random.seed(0)
    # x = np.random.randn(n)
    # x = np.array([-2, -1]).reshape(n, 1)
    # xxT = np.outer(x, x)
    # print(x, xxT)

    x = cp.Variable((n, 1))  # Define as 2d array to simplify bmat
    xxT = cp.Variable((n, n), symmetric=True)

    A = np.random.randn(n, n)
    A = (A + A.T) / 2
    print("A = \n", A)

    l = 4 * np.ones(n)
    l = l.reshape(n, 1)
    u = 5 * np.ones(n)
    u = u.reshape(n, 1)

    z = cp.Variable((n, 1))  # Define as 2d array to simplify bmat
    zzT = cp.Variable((n, n), symmetric=True)
    zyT = cp.Variable((n, n))
    zxT = cp.Variable((n, n))

    y = cp.Variable((n, 1))
    yyT = cp.Variable((n, n), symmetric=True)
    yxT = cp.Variable((n, n))

    obj = cp.Maximize(cp.trace(A @ zzT))
    P = cp.Variable((3 * n + 1, 3 * n + 1), symmetric=True)
    constraints = [y >= l, y >= x, cp.diag(yyT - yxT - l @ y.T + l @ x.T) == 0,
                   z <= u, z <= y, cp.diag(u @ y.T - u @ z.T - zyT + zzT) == 0,
                   cp.sum_squares(x) <= R ** 2, cp.trace(xxT) <= R ** 2,
                   P >> 0]

    constraints += [P == cp.bmat([[zzT, zyT, zxT, z],
                                  [zyT.T, yyT, yxT, y],
                                  [zxT.T, yxT.T, xxT, x],
                                  [z.T, y.T, x.T, np.array([[1]])]])]

    prob = cp.Problem(obj, constraints)
    res = prob.solve()
    print("objective = ", res)
    print("P = \n", np.round(P.value, 4))
    print('z = ', np.round(P.value[0:2, -1]))


def main():
    # test_box_proj_gurobi()
    # test_relu_SDR_blocks()
    test_box_proj_SDR()


if __name__ == '__main__':
    main()
