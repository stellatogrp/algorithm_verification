import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp
from qcqp2quad_form.quad_extractor import QuadExtractor
from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from quadratic_functions.extended_slemma_sdp import *

from gurobipy import GRB


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
    print(A)

    x = cp.Variable(n)
    xxT = cp.Variable((n, n), symmetric=True)

    z = cp.Variable(n)
    zzT = cp.Variable((n, n), symmetric=True)
    zxT = cp.Variable((n, n))

    obj = cp.Maximize(cp.trace(A @ zzT))
    P = cp.Variable((2 * n + 1, 2 * n + 1), symmetric=True)
    constraints = [z >= x, z >= 0, cp.diag(zzT) == cp.diag(zxT), P >> 0,
                   cp.sum_squares(x) <= R ** 2, cp.trace(xxT) <= R ** 2]

    # blocks = cp.bmat([[zzT, zxT, z], [zxT.T, xxT, x], [z.T, x.T, 1]])
    blocks = cp.bmat([[zzT, zxT, cp.reshape(z, (n, 1))],
                      [zxT.T, xxT, cp.reshape(x, (n, 1))],
                      # [cp.reshape(z, (1, n)), cp.reshape(x, (1, n)), 1]
                      ])

    blocks_bottom_row = cp.reshape(cp.hstack([z, x, 1]), (1, 2 * n + 1))
    constraints.append(P == cp.vstack([blocks, blocks_bottom_row]))

    prob = cp.Problem(obj, constraints)
    res = prob.solve()
    print(res)
    print(np.round(P.value, 4))


def test_box_proj_SDR():
    print('--------testing SDR--------')
    n = 6
    np.random.seed(0)
    x = np.random.randn(n)
    print(x)
    l = -.5
    u = 1


def main():
    test_box_proj_gurobi()
    test_relu_SDR_blocks()
    # test_box_proj_SDR()


if __name__ == '__main__':
    main()
