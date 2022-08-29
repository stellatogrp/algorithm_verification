import cvxpy as cp
import numpy as np

from certification_problem.basic_algorithm_steps.linear_step import LinearStep


def max_vec_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars, add_RLT):
    step = steps[i]
    prev_step = steps[i-1]

    y = step.get_output_var()
    x = step.get_input_var()
    l = step.get_lower_bound_vec()
    l = l.reshape(-1, 1)

    y_var = curr.iterate_vars[y].get_cp_var()
    yyT_var = curr.iterate_outerproduct_vars[y]
    x_var = curr.iterate_vars[x].get_cp_var()
    xxT_var = curr.iterate_outerproduct_vars[x]

    yxT_var = curr.iterate_cross_vars[y][x]

    constraints = [y_var >= l, y_var >= x_var,
                   cp.diag(yyT_var) >= cp.diag(l @ l.T),
                   cp.diag(yyT_var - yxT_var - l @ y_var.T + l @ x_var.T) == 0]
    constraints += [
        cp.bmat([
            [yyT_var, yxT_var, y_var],
            [yxT_var.T, xxT_var, x_var],
            [y_var.T, x_var.T, np.array([[1]])]
        ]) >> 0,
    ]

    if type(prev_step) == LinearStep:
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
