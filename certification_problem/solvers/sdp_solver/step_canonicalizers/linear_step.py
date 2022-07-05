# TODO: Check https://github.com/cvxpy/cvxpy/blob/fb1f271b94edcbfe5f02fa4bd93ef6bb9fbdd269/cvxpy/reductions/eliminate_pwl/atom_canonicalizers/abs_canon.py
import cvxpy as cp
import numpy as np

from certification_problem.algorithm_steps.block_step import BlockStep


def linear_step_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    step = steps[i]
    prev_step = steps[i-1]
    u = step.get_rhs_var()
    y = step.get_output_var()
    A = step.get_rhs_matrix()
    D = step.get_lhs_matrix()
    b = step.get_rhs_const_vec()
    y_var = curr.iterate_vars[y]
    yyT_var = curr.iterate_outerproduct_vars[y]
    u_var = curr.iterate_vars[u]
    uuT_var = curr.iterate_outerproduct_vars[u]
    yuT_var = curr.iterate_cross_vars[y][u]
    # constraints = [y_var == A @ u_var, yyT_var == A @ uuT_var @ A.T, yuT_var == A @ uuT_var]
    constraints = [D @ y_var == A @ u_var + b,
                   D @ yyT_var @ D.T == A @ uuT_var @ A.T + A @ u_var @ b.T + b @ u_var.T @ A.T + b @ b.T,
                   D @ yuT_var == A @ uuT_var + b @ u_var.T,
                   D @ yuT_var @ A.T + D @ y_var @ b.T == D @ yyT_var @ D.T]

    yuT_blocks = []
    if type(prev_step) == BlockStep:  # this should always be true
        block_vars = prev_step.list_x
        for var in block_vars:
            if var.is_param:
                yuT_blocks.append(curr.iterate_param_vars[y][var])
            else:
                yuT_blocks.append(curr.iterate_cross_vars[y][var])
        constraints += [
            yuT_var == cp.bmat([
                yuT_blocks
            ])
        ]

    constraints += [
        cp.bmat([
            [yyT_var, yuT_var, y_var],
            [yuT_var.T, uuT_var, u_var],
            [y_var.T, u_var.T, np.array([[1]])]
        ]) >> 0,
    ]
    return constraints
