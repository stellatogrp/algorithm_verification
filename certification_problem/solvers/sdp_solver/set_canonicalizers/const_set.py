import cvxpy as cp
import numpy as np


def const_canon(init_set, handler):
    x = init_set.get_iterate()
    val = init_set.val
    x_var = handler.iterate_vars[x].get_cp_var()
    xxT_var = handler.iterate_outerproduct_vars[x]
    return [
                x_var == val,
                xxT_var == val @ val.T,
           ]
