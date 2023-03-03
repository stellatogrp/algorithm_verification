import cvxpy as cp
import numpy as np

# from algocert.solvers.sdp_solver.var_bounds.RLT_constraints import \
#     RLT_constraints


def l1_conv_resid_canon(obj, handlers, add_RLT):
    constraints = []

    K = len(handlers) - 1
    handlerK = handlers[K]
    handlerKminus1 = handlers[K - 1]
    x = obj.get_iterate()
    n = x.get_dim()

    xK = handlerK.iterate_vars[x].get_cp_var()
    xKminus1 = handlerKminus1.iterate_vars[x].get_cp_var()
    xKxKT = handlerK.iterate_outerproduct_vars[x]
    xKminus1xKminus1T = handlerKminus1.iterate_outerproduct_vars[x]
    xcross = handlerK.iterate_cross_vars[x][x]

    # xN = handlerN.iterate_vars[x].get_cp_var()
    # xNxNT = handlerN.iterate_outerproduct_vars[x]
    # xNminus1 = handlerNminus1.iterate_vars[x].get_cp_var()
    # xNminus1xNminus1T = handlerNminus1.iterate_outerproduct_vars[x]
    # xcross = handlerN.iterate_cross_vars[x][x]
    # ret_obj = cp.trace(xNxNT - 2 * xcross + xNminus1xNminus1T)

    t = xK - xKminus1
    # Goal, y = (t)_+ and z = (-t)_+, which means |t| = y + z

    y = cp.Variable((n, 1))
    z = cp.Variable((n, 1))

    yyT = cp.Variable((n, n))
    yzT = cp.Variable((n, n))
    ytT = cp.Variable((n, n))

    zzT = cp.Variable((n, n))
    ztT = cp.Variable((n, n))
    ttT = xKxKT - 2 * xcross + xKminus1xKminus1T

    constraints = [
        y >= 0, yyT >= 0, y >= t, cp.diag(yyT - ytT) == 0,
        z >= 0, zzT >= 0, z >= -t, cp.diag(zzT + ztT) == 0,
        cp.bmat([
            [yyT, yzT, ytT, y],
            [yzT.T, zzT, ztT, z],
            [ytT.T, ztT.T, ttT, t],
            [y.T, z.T, t.T, np.array([[1]])]
        ]) >> 0,

        # z <= 0, zzT >= 0, z <= t, cp.diag(zzT - ztT) == 0,
        # cp.bmat([
        #     [yyT, yzT, ytT, y],
        #     [yzT.T, zzT, ztT, z],
        #     [ytT.T, ztT.T, ttT, t],
        #     [y.T, z.T, t.T, np.array([[1]])]
        # ]) >> 0,

    ]

    return cp.sum(y+z), constraints
