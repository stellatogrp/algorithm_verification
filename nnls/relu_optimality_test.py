import numpy as np
import scipy.sparse as spa
import gurobipy as gp

from gurobipy import GRB


def test_relu_KKT():
    n = 10
    y_val = np.random.randn(n)

    m = gp.Model()
    m.setParam('NonConvex', 2)

    z = m.addMVar(n, name='z')  # default [0. inf) constraints are necessary
    gamma = m.addMVar(n, name='gamma')

    m.setObjective(0)
    m.addConstr(z - y_val - gamma == 0)
    m.addConstr(gamma @ z == 0)
    m.optimize()

    print('original y:', np.round(y_val, 4))
    print('solution to QCQP:', np.round(z.X, 4))


def main():
    test_relu_KKT()


if __name__ == '__main__':
    main()
