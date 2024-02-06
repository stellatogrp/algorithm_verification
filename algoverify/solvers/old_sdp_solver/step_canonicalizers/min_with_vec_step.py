import cvxpy as cp
import numpy as np

from algoverify.basic_algorithm_steps.basic_linear_step import BasicLinearStep
from algoverify.variables.parameter import Parameter


def min_vec_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars, add_RLT, kwargs):
    step = steps[i]
    prev_step = steps[i-1]

    y = step.get_output_var()
    x = step.get_input_var()
    u = step.get_upper_bound_vec()

    y_var = curr.iterate_vars[y].get_cp_var()
    yyT_var = curr.iterate_outerproduct_vars[y]
    x_var = curr.iterate_vars[x].get_cp_var()
    xxT_var = curr.iterate_outerproduct_vars[x]

    yxT_var = curr.iterate_cross_vars[y][x]

    if not type(u) == Parameter:
        u_vec = u.reshape(-1, 1)
        constraints = [y_var <= u_vec, y_var <= x_var,
                       # cp.diag(yyT_var) <= cp.diag(u @ u.T),
                       cp.diag(yyT_var - yxT_var - u_vec @ y_var.T + u_vec @ x_var.T) == 0]

        constraints += [
            cp.bmat([
                [yyT_var, yxT_var, y_var],
                [yxT_var.T, xxT_var, x_var],
                [y_var.T, x_var.T, np.array([[1]])]
            ]) >> 0,
        ]

    else:
        u_var = param_vars[u].get_cp_var()
        uuT_var = param_outerproduct_vars[u]
        yuT_var = curr.iterate_param_vars[y][u]
        xuT_var = curr.iterate_param_vars[x][u]
        constraints = [y_var <= u_var, y_var <= x_var,
                       # cp.diag(yyT_var) <= cp.diag(uuT_var),
                       cp.diag(yyT_var - yxT_var - yuT_var + xuT_var) == 0]

        constraints += [
            cp.bmat([
                [yyT_var, yxT_var, yuT_var, y_var],
                [yxT_var.T, xxT_var, xuT_var, x_var],
                [yuT_var.T, xuT_var.T, uuT_var, u_var],
                [y_var.T, x_var.T, u_var.T, np.array([[1]])]
            ]) >> 0,
        ]

    if type(prev_step) == BasicLinearStep:
        # print(type(x))
        A = prev_step.get_rhs_matrix()
        D = prev_step.get_lhs_matrix()
        b = prev_step.get_rhs_const_vec()
        b = b.reshape(-1, 1)
        block_step = steps[i-2]
        u = block_step.get_output_var()
        u_var = curr.iterate_vars[u].get_cp_var()
        uuT_var = curr.iterate_outerproduct_vars[u]
        block_vars = block_step.list_x
        # print(block_vars)
        yuT_var = curr.iterate_cross_vars[y][u]
        yuT_blocks = []
        for var in block_vars:
            if var.is_param:
                yuT_blocks.append(curr.iterate_param_vars[y][var])
            else:
                yuT_blocks.append(curr.iterate_cross_vars[y][var])
        constraints += [
            yuT_var == cp.bmat([
                yuT_blocks
            ]),
            cp.bmat([
                [yyT_var, yuT_var, y_var],
                [yuT_var.T, uuT_var, u_var],
                [y_var.T, u_var.T, np.array([[1]])]
            ]) >> 0,
            yxT_var @ D.T == yuT_var @ A.T + y_var @ b.T,
        ]

    return constraints


def min_vec_bound_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    step = steps[i]
    y = step.get_output_var()
    x = step.get_input_var()
    u = step.get_upper_bound_vec()
    lower_x = curr.iterate_vars[x].get_lower_bound()
    upper_x = curr.iterate_vars[x].get_upper_bound()
    if not type(u) == Parameter:
        lower_u = u
        upper_u = u
    else:
        lower_u = param_vars[u].get_lower_bound()
        upper_u = param_vars[u].get_upper_bound()
    lower_y = np.minimum(lower_x, lower_u)
    upper_y = np.minimum(upper_x, upper_u)
    curr.iterate_vars[y].set_lower_bound(lower_y)
    curr.iterate_vars[y].set_upper_bound(upper_y)
