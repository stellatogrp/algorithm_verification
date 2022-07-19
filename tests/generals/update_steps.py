import cvxpy as cp
import numpy as np


def linear_update(u, C):
    '''
    Return a new variable y with constraints that enforce SDP relaxations of y = Cu and yy^T = Cu(uC)^T
    '''
    n = u.shape[0]
    y = cp.Variable((n, 1))
    yyT = cp.Variable((n, n))
    constraints = [C @ y == 0, C @ yyT @ C.T == 0,
                   cp.bmat([
                       [yyT, y],
                       [y.T, np.array([[1]])]
                   ]) >> 0]
    return y, yyT, constraints
