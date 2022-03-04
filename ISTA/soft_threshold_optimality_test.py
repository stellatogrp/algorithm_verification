import numpy as np
import scipy.sparse as spa
import gurobipy as gp

from gurobipy import GRB


def test_soft_threshold():
    lambdat = .1
    y_val = np.array([-1, -.05, 0, .05, 1])
    n = len(y_val)
    I = spa.eye(n)
    halfI = .5 * I
    minusI = -1 * I
    lambdat_scaled_ones = lambdat * np.ones(n)

    m = gp.Model()
    m.setParam('NonConvex', 2)

    # y = m.setParam('y', y_val)
    y = m.addMVar(n,
                  name='y',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))
    m.addConstr(y == y_val)
    # y = y_val
    v = m.addMVar(n,
                  name='v',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))

    u = m.addMVar(n, name='u')  # default [0. inf) constraints are sufficient

    obj = lambdat_scaled_ones @ u + v @ halfI @ v + v @ minusI @ y + y @ halfI @ y
    m.setObjective(obj)
    m.addConstr(v - u <= 0)
    m.addConstr(-v - u <= 0)
    m.optimize()
    # import pdb
    # pdb.set_trace()
    print(v.X, u.X)

    # get a variable value like gamma1.X


def test_soft_threshold_KKT():
    '''
        Set up the QCQP to solve the feasibility problem to check if the prox of the l1 function can be
            represented implicitly, i.e. if the feasibility problem with the KKT conditions has the solution
            corresponding to the soft_threshold function
    '''
    lambdat = .2
    # y_val = np.array([-1, -.05, 0, .05, 1])
    # n = len(y_val)

    n = 10
    y_val = np.random.randn(n)

    I = spa.eye(n)
    halfI = .5 * I
    minusI = -1 * I
    lambdat_scaled_ones = lambdat * np.ones(n)

    m = gp.Model()
    m.setParam('NonConvex', 2)

    # y = m.setParam('y', y_val)
    y = m.addMVar(n,
                  name='y',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))
    m.addConstr(y == y_val)
    # y = y_val
    v = m.addMVar(n,
                  name='v',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))

    u = m.addMVar(n, name='u')  # default [0. inf) constraints are sufficient
    # gamma1 = m.addMVar(n,
    #                    name='gamma1',
    #                    ub=gp.GRB.INFINITY * np.ones(n),
    #                    lb=-gp.GRB.INFINITY * np.ones(n))
    # gamma2 = m.addMVar(n,
    #                    name='gamma2',
    #                    ub=gp.GRB.INFINITY * np.ones(n),
    #                    lb=-gp.GRB.INFINITY * np.ones(n))

    # using the fact that gurobi defaults to [0. inf) constraints, we can just leave these as is
    gamma1 = m.addMVar(n, name='gamma1')
    gamma2 = m.addMVar(n, name='gamma2')

    m.setObjective(0)
    m.addConstr(v - y + gamma1 - gamma2 == 0)
    m.addConstr(lambdat_scaled_ones - gamma1 - gamma2 == 0)
    # m.addConstr(gamma1 >= 0)
    # m.addConstr(gamma2 >= 0)
    m.addConstr(v - u <= 0)
    m.addConstr(-v - u <= 0)
    m.addConstr(gamma1 @ v - gamma1 @ u == 0)
    m.addConstr(gamma2 @ v + gamma2 @ u == 0)
    m.optimize()
    # import pdb
    # pdb.set_trace()

    print('original y:', np.round(y_val, 4))
    print('threshold used:', lambdat)
    print('solution to QCQP:', np.round(v.X, 4))

    # get a variable value like gamma1.X


def main():
    # test_soft_threshold()
    test_soft_threshold_KKT()


if __name__ == '__main__':
    main()
