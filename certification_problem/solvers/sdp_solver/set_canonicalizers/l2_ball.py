import cvxpy as cp
import numpy as np


def l2_ball_canon(init_set, handler):
    # TODO: actually incorporate the center
    x = init_set.get_iterate()
    r = init_set.r
    x_var = handler.iterate_vars[x]
    xxT_var = handler.iterate_outerproduct_vars[x]
    return [cp.sum_squares(x_var) <= r ** 2, cp.trace(xxT_var) <= r ** 2,
                cp.bmat([
                    [xxT_var, x_var],
                    [x_var.T, np.array([[1]])]
                ]) >> 0,
            ]
