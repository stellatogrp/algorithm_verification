import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp

from gurobipy import GRB


def form_problem_and_extract_data():
    n = 5
    alpha = 1
    beta = 0
    I = np.eye(n)
    np.random.seed(1)
    x = np.random.randn(n)
    print('original y:', np.round(x, 4))

    obj = 0
    constraints = []

    y = cp.Variable(n)
    z = cp.hstack([x, y, 1])
    lambd = cp.Variable((n, n))
    nu = cp.Variable(n)
    eta = cp.Variable(n)
    T = 0

    for i in range(n):
        for j in range(i+1, n):
            T += lambd[i, j] * np.outer(I[i] - I[j], I[i] - I[j])

    lambdaT = (cp.diag(cp.diag(lambd)) + T)
    Q11 = -2 * alpha * beta * lambdaT
    Q12 = (alpha + beta) * lambdaT
    Q13 = -beta * nu - alpha * eta
    Q22 = -2 * lambdaT
    Q23 = nu + eta
    Q33 = 0

    # Q = cp.bmat([[Q11, Q12, Q13], [Q12.T, Q22, Q23], [Q13.T, Q23.T, Q33]])
    Q = cp.bmat([[Q11, Q12], [Q12.T, Q22]])


def test_relu_QC():
    n = 5
    I = np.eye(n)
    np.random.seed(1)
    y_val = np.random.randn(n)
    print('original y:', np.round(y_val, 4))

    m = gp.Model()
    m.setParam('NonConvex', 2)

    lambd = m.addMVar((n, n), name='lambd')
    T = m.addMVar((n, n),
                  name='T',
                  ub=gp.GRB.INFINITY * np.ones((n, n)),
                  lb=-gp.GRB.INFINITY * np.ones((n, n)))


def main():
    # test_relu_QC()
    form_problem_and_extract_data()


if __name__ == '__main__':
    main()
