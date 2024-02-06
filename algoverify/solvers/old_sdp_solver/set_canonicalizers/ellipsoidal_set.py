import cvxpy as cp
import numpy as np


def ellipsoidal_canon(init_set, handler):
    x = init_set.get_iterate()
    Q = init_set.Q
    c = init_set.c
    x_var = handler.iterate_vars[x].get_cp_var()
    xxT_var = handler.iterate_outerproduct_vars[x]
    return [
        cp.quad_form(x_var, Q) - 2 * (c.T @ Q) @ x_var + c.T @ Q @ c <= 1,
        cp.trace(Q @ xxT_var) - 2 * (c.T @ Q) @ x_var <= 1,
        cp.bmat([
            [xxT_var, x_var],
            [x_var.T, np.array([[1]])]
        ]) >> 0,
    ]
