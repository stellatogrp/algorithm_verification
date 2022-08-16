import cvxpy as cp
import numpy as np


from certification_problem.solvers.sdp_solver.var_bounds.RLT_constraints import RLT_constraints


def conv_resid_canon(obj, handlers, add_RLT):
    constraints = []

    N = len(handlers) - 1
    handlerN = handlers[N]
    handlerNminus1 = handlers[N - 1]
    x = obj.get_iterate()
    n = x.get_dim()
    xN = handlerN.iterate_vars[x].get_cp_var()
    xNxNT = handlerN.iterate_outerproduct_vars[x]
    xNminus1 = handlerNminus1.iterate_vars[x].get_cp_var()
    xNminus1xNminus1T = handlerNminus1.iterate_outerproduct_vars[x]
    xcross = handlerN.iterate_cross_vars[x][x]
    ret_obj = cp.trace(xNxNT - 2 * xcross + xNminus1xNminus1T)
    # constraints = [
    #     cp.bmat([
    #         [xNxNT, xcross, xN],
    #         [xcross.T, xNminus1xNminus1T, xNminus1],
    #         [xN.T, xNminus1.T, np.array([[1]])]
    #     ]) >> 0,
    # ]

    if add_RLT:
        lower_xN = handlerN.iterate_vars[x].get_lower_bound()
        upper_xN = handlerN.iterate_vars[x].get_upper_bound()
        lower_xNminus1 = handlerNminus1.iterate_vars[x].get_lower_bound()
        upper_xNminus1 = handlerNminus1.iterate_vars[x].get_upper_bound()
        constraints += RLT_constraints(xcross, xN, lower_xN, upper_xN, xNminus1, lower_xNminus1, upper_xNminus1)

    test_x = True
    if test_x:
        for i in range(N):
            curr = handlers[i+1]
            prev = handlers[i]
            xi = curr.iterate_vars[x].get_cp_var()
            lower_xi = curr.iterate_vars[x].get_lower_bound()
            upper_xi = curr.iterate_vars[x].get_upper_bound()
            lower_ximinus1 = prev.iterate_vars[x].get_lower_bound()
            upper_ximinus1 = prev.iterate_vars[x].get_upper_bound()
            xixiT = curr.iterate_outerproduct_vars[x]
            ximinus1 = prev.iterate_vars[x].get_cp_var()
            ximinus1x1minus1T = prev.iterate_outerproduct_vars[x]
            xcross = curr.iterate_cross_vars[x][x]
            constraints += [
                cp.bmat([
                    [xixiT, xcross, xi],
                    [xcross.T, ximinus1x1minus1T, ximinus1],
                    [xi.T, ximinus1.T, np.array([[1]])]
                ]) >> 0,
            ]
            if add_RLT:
                constraints += RLT_constraints(xcross, xi, lower_xi, upper_xi, ximinus1, lower_ximinus1, upper_ximinus1)
    # c = np.ones(n)
    # ret_obj = cp.trace(xNxNT)
    return ret_obj, constraints
