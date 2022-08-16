import cvxpy as cp
import numpy as np

from certification_problem.basic_algorithm_steps.linear_step import LinearStep
from certification_problem.solvers.sdp_solver.var_bounds.RLT_constraints import RLT_constraints


def nonneg_orthant_proj_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars, add_RLT):
    step = steps[i]
    prev_step = steps[i-1]

    y = step.get_output_var()
    x = step.get_input_var()
    n = x.get_dim()

    y_var = curr.iterate_vars[y].get_cp_var()
    yyT_var = curr.iterate_outerproduct_vars[y]
    x_var = curr.iterate_vars[x].get_cp_var()
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

    if type(prev_step) == LinearStep:
        # print(type(x))
        A = prev_step.get_rhs_matrix()
        D = prev_step.get_lhs_matrix()
        b = prev_step.get_rhs_const_vec()
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

    lower_y = curr.iterate_vars[y].get_lower_bound()
    upper_y = curr.iterate_vars[y].get_upper_bound()
    lower_x = curr.iterate_vars[x].get_lower_bound()
    upper_x = curr.iterate_vars[x].get_upper_bound()
    if add_RLT:
        extra_constraints = RLT_constraints(yxT_var, y_var, lower_y, upper_y, x_var, lower_x, upper_x)
        constraints += extra_constraints

        # constraints += RLT_constraints(xxT_var, x_var, lower_x, upper_x, x_var, lower_x, upper_x)
        # constraints += [lower_x <= x_var, x_var <= upper_x]
        # constraints += RLT_constraints(yyT_var, y_var, lower_y, upper_y, y_var, lower_y, upper_y)
        # constraints += [lower_y <= y_var, y_var <= upper_y]
        # triangle constraints
        print(upper_y, lower_y, upper_x, lower_x)
        frac = np.divide(upper_y - lower_y, upper_x - lower_x)
        print('frac', frac)
        A = np.zeros((n, n))
        for i in range(n):
            # this aboslutely should not have been necessary, but for some reason I couldn't get the numpy functions
            # to work as desired
            A[i, i] = frac[i, 0]
        b = np.multiply(frac, -lower_x) + lower_y
        constraints += [
            y_var <= A @ x_var + b,
            yyT_var <= A @ xxT_var @ A.T + A @ x_var @ b.T + b @ x_var.T @ A.T + b @ b.T
        ]
    return constraints


def nonneg_orthant_proj_bound_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    step = steps[i]
    prev_step = steps[i - 1]
    y = step.get_output_var()
    x = step.get_input_var()
    n = y.get_dim()
    lower_x = curr.iterate_vars[x].get_lower_bound()
    upper_x = curr.iterate_vars[x].get_upper_bound()
    # print(lower_x, upper_x)
    zeros = np.zeros((n, 1))
    lower_y = np.maximum(lower_x, zeros)
    upper_y = np.maximum(upper_x, zeros)
    # print(lower_y, upper_y)
    curr.iterate_vars[y].set_lower_bound(lower_y)
    curr.iterate_vars[y].set_upper_bound(upper_y)
