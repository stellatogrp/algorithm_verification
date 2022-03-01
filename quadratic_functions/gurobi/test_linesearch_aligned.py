import numpy as np
import gurobipy as gp

from gurobipy import GRB


def test_basic_pep():
    n = 2
    mu = 2
    L = 20
    R = 1
    gamma = 2 / (mu + L)

    I = np.eye(n)
    Z = np.zeros((n, n))
    P = np.array([[mu, 0], [0, L]])

    m = gp.Model()
    m.params.NonConvex = 2
    m.setParam('NonConvex', 2)

    x = m.addMVar(n, name='x')
    m.setObjective((x @ (gamma ** 2 * P @ P - gamma * 2 * P + I) @ x), GRB.MAXIMIZE)
    m.addConstr(x @ I @ x <= 1)
    m.optimize()  # should be ((L-mu)/(L+mu)) ** 2


def test_linesearch_aligned():
    n = 2
    mu = 2
    L = 20
    R = 1

    I = np.eye(n)
    Z = np.zeros((n, n))
    P = np.array([[mu, 0], [0, L]])

    Hobj = .5 * np.block([[Z, Z], [Z, P]])  # not negative since we maximize
    cobj = np.zeros(2 * n)
    dobj = 0

    # Hineq = .5 * np.block([[P, Z], [Z, Z]])
    Hineq = np.block([[I, Z], [Z, Z]])

    Heq1 = np.block([[Z, -.5 * P], [-.5 * P.T, P]])

    Heq2 = np.block([[Z, .5 * P @ P], [.5 * P @ P, Z]])

    m = gp.Model()
    m.params.NonConvex = 2
    m.setParam('NonConvex', 2)

    y = m.addMVar(2 * n, name='y')
    m.setObjective(y @ Hobj @ y, GRB.MAXIMIZE)
    m.addConstr(y @ Hineq @ y <=  R ** 2)
    m.addConstr(y @ Heq1 @ y == 0)
    m.addConstr(y @ Heq2 @ y == 0)

    m.optimize()

    # m.printAttr('y')


def main():
    # test_basic_pep()
    test_linesearch_aligned()


if __name__ == '__main__':
    main()
