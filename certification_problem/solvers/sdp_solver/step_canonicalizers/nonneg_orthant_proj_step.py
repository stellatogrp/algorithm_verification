import cvxpy as cp
import numpy as np


def nonneg_orthant_proj_canon(step, curr, prev, iter_id_map):
    y = step.get_output_var()
    x = step.get_input_var()

    y_var = curr.iterate_vars[y]
    yyT_var = curr.iterate_outerproduct_vars[y]
    x_var = curr.iterate_vars[x]
    xxT_var = curr.iterate_outerproduct_vars[x]

    yxT_var = curr.iterate_cross_vars[y][x]
    constraints = [y_var >= 0, yyT_var >= 0, y_var >= x_var, cp.diag(yyT_var - yxT_var) == 0]
    constraints += [
        cp.bmat([
            [yyT_var, yxT_var, y_var],
            [yxT_var.T, xxT_var, x_var],
            [y_var.T, x_var.T, np.array([[1]])]
        ]) >> 0,
    ]
    return constraints
