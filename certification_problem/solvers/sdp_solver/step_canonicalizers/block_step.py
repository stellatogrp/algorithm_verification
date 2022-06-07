import cvxpy as cp
import numpy as np


def block_step_canon(step, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    print(step.list_x)
    u = step.get_output_var()
    u_var = curr.iterate_vars[u]
    uuT_var = curr.iterate_outerproduct_vars[u]

    return [cp.sum_squares(uuT_var) <= 1,
            cp.bmat([
                [uuT_var, u_var],
                [u_var.T, np.array([[1]])]
            ]) >> 0,
            ]
