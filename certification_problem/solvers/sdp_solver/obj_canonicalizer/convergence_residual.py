import cvxpy as cp
import numpy as np


def conv_resid_canon(obj, handlerN, handlerNminus1):
    x = obj.get_iterate()
    xN = handlerN.iterate_vars[x].get_cp_var()
    xNxNT = handlerN.iterate_outerproduct_vars[x]
    xNminus1 = handlerNminus1.iterate_vars[x].get_cp_var()
    xNminus1xNminus1T = handlerNminus1.iterate_outerproduct_vars[x]
    xcross = handlerN.iterate_cross_vars[x][x]
    ret_obj = cp.trace(xNxNT - 2 * xcross + xNminus1xNminus1T)
    constraints = [
        cp.bmat([
            [xNxNT, xcross, xN],
            [xcross.T, xNminus1xNminus1T, xNminus1],
            [xN.T, xNminus1.T, np.array([[1]])]
        ]) >> 0,
    ]
    return ret_obj, constraints
