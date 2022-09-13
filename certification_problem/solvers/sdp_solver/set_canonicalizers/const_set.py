import cvxpy as cp
import numpy as np


def const_canon(init_set, handler):
    x = init_set.get_iterate()
    val = init_set.val
    val = val.reshape(-1, 1)
    x_var = handler.iterate_vars[x].get_cp_var()
    xxT_var = handler.iterate_outerproduct_vars[x]
    return [
                x_var == val,
                xxT_var == val @ val.T,
           ]


def const_bound_canon(init_set, handler):
    x = init_set.get_iterate()
    val = init_set.val
    handler.iterate_vars[x].set_lower_bound(val)
    handler.iterate_vars[x].set_upper_bound(val)
