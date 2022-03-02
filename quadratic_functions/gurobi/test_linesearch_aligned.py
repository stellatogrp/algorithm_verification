import numpy as np
import scipy.sparse as spa
import gurobipy as gp

from gurobipy import GRB


def test_basic_pep():
    n = 2
    mu = 2
    L = 20
    #  R = 1
    gamma = 2 / (mu + L)

    I = spa.eye(n)
    P = spa.csc_matrix(np.array([[mu, 0], [0, L]]))

    m = gp.Model()
    m.setParam('NonConvex', 2)

    x = m.addMVar(n, name='x',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))
    m.setObjective((x @ (gamma ** 2 * P @ P - gamma * 2 * P + I) @ x),
                   GRB.MAXIMIZE)
    m.addConstr(x @ I @ x <= 1)
    m.optimize()  # should be ((L-mu)/(L+mu)) ** 2


def test_linesearch_aligned():
    n = 2
    mu = 2
    L = 20
    R = 1

    I = spa.eye(n)
    Z = spa.csc_matrix((n, n))
    P = spa.csc_matrix([[mu, 0], [0, L]])

    Hobj = -.5 * spa.bmat([[Z, Z], [Z, P]], format='csc')
    #  cobj = np.zeros(2 * n)
    #  dobj = 0

    # Set up distance constraint on x0
    Hineq = spa.bmat([[I, Z], [Z, Z]], format='csc')

    # Set up two equality constraints for line search iterates
    Heq1 = spa.bmat([[Z, -.5 * P],
                     [-.5 * P.T, P]], format='csc')
    Heq2 = spa.bmat([[Z, .5 * P @ P],
                     [.5 * P @ P, Z]], format='csc')

    m = gp.Model()
    m.setParam('NonConvex', 2)

    y = m.addMVar(2 * n,
                  name='y',
                  ub=gp.GRB.INFINITY * np.ones(2 * n),
                  lb=-gp.GRB.INFINITY * np.ones(2 * n))
    m.setObjective(y @ Hobj @ y, GRB.MINIMIZE)
    m.addConstr(y @ Hineq @ y <= R ** 2)
    m.addConstr(y @ Heq1 @ y == 0)
    m.addConstr(y @ Heq2 @ y == 0)
    m.addConstr(y @ Hobj @ y <= -.75)  # fake constraint to give gurobi an upper bound

    m.optimize()
    # import pdb
    # pdb.set_trace()
    y_grb = m.getVars()
    y = np.array([y_grb[i].X for i in range(2 * n)])
    print(y)
    # m.printAttr('y')


def main():
    #  test_basic_pep()
    test_linesearch_aligned()


if __name__ == '__main__':
    main()
