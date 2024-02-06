import cvxpy as cp

from algoverify.solvers.sdp_solver.var_bounds.RLT_constraints import RLT_constraints


def outer_prod_trace_canon(obj, handlers, add_RLT):
    constraints = []

    N = len(handlers) - 1
    handlerN = handlers[N]
    handlerNminus1 = handlers[N - 1]
    x = obj.get_iterate()
    #  n = x.get_dim()
    xN = handlerN.iterate_vars[x].get_cp_var()
    xNxNT = handlerN.iterate_outerproduct_vars[x]
    xNminus1 = handlerNminus1.iterate_vars[x].get_cp_var()
    #  xNminus1xNminus1T = handlerNminus1.iterate_outerproduct_vars[x]
    xcross = handlerN.iterate_cross_vars[x][x]

    if add_RLT:
        lower_xN = handlerN.iterate_vars[x].get_lower_bound()
        upper_xN = handlerN.iterate_vars[x].get_upper_bound()
        lower_xNminus1 = handlerNminus1.iterate_vars[x].get_lower_bound()
        upper_xNminus1 = handlerNminus1.iterate_vars[x].get_upper_bound()
        constraints += RLT_constraints(xcross, xN, lower_xN, upper_xN, xNminus1, lower_xNminus1, upper_xNminus1)

    return cp.trace(xNxNT), constraints
