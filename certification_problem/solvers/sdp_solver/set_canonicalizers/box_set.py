import cvxpy as cp
import numpy as np


def box_canon(init_set, handler):
    x = init_set.get_iterate()
    n = x.get_dim()
    l = init_set.l
    u = init_set.u
    x_var = handler.iterate_vars[x].get_cp_var()
    xxT_var = handler.iterate_outerproduct_vars[x]
    return [
                # l <= x_var <= u,
                # cp.reshape(cp.diag(xxT_var), (n, 1)) <= (l + u) * x_var - l * u,
                cp.reshape(cp.diag(xxT_var), (n, 1)) <= cp.multiply((l + u), x_var) - l * u,
           ]


def box_bound_canon(init_set, handler):
    x = init_set.get_iterate()
    n = x.get_dim()
    l = init_set.l
    u = init_set.u
    handler.iterate_vars[x].set_lower_bound(l)
    handler.iterate_vars[x].set_upper_bound(u)
