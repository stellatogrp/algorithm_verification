# TODO: Check https://github.com/cvxpy/cvxpy/blob/fb1f271b94edcbfe5f02fa4bd93ef6bb9fbdd269/cvxpy/reductions/eliminate_pwl/atom_canonicalizers/abs_canon.py
import cvxpy as cp
import numpy as np

from certification_problem.basic_algorithm_steps.block_step import BlockStep
from certification_problem.solvers.sdp_solver.var_bounds.RLT_constraints import RLT_constraints


def linear_step_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars, add_RLT):
    step = steps[i]
    prev_step = steps[i-1]
    u = step.get_rhs_var()
    y = step.get_output_var()
    A = step.get_rhs_matrix()
    D = step.get_lhs_matrix()
    b = step.get_rhs_const_vec()
    b = b.reshape(-1, 1)
    y_var = curr.iterate_vars[y].get_cp_var()
    yyT_var = curr.iterate_outerproduct_vars[y]
    u_var = curr.iterate_vars[u].get_cp_var()
    uuT_var = curr.iterate_outerproduct_vars[u]
    yuT_var = curr.iterate_cross_vars[y][u]
    # constraints = [y_var == A @ u_var, yyT_var == A @ uuT_var @ A.T, yuT_var == A @ uuT_var]
    # print(A.shape, u_var.shape, b.shape)
    # print(A @ u_var @ b.T)
    # exit(0)
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

    lower_y = curr.iterate_vars[y].get_lower_bound()
    upper_y = curr.iterate_vars[y].get_upper_bound()
    lower_u = curr.iterate_vars[u].get_lower_bound()
    upper_u = curr.iterate_vars[u].get_upper_bound()
    if add_RLT:
        constraints += RLT_constraints(yuT_var, y_var, lower_y, upper_y, u_var, lower_u, upper_u)

    # bound prop
    # constraints += RLT_constraints(yyT_var, y_var, lower_y, upper_y, y_var, lower_y, upper_y)
    # constraints += [lower_y <= y_var, y_var <= upper_y]

    return constraints


def linear_step_bound_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    step = steps[i]
    prev_step = steps[i - 1]
    u = step.get_rhs_var()
    y = step.get_output_var()
    A = step.get_rhs_matrix()
    D = step.get_lhs_matrix()
    Dinv = step.get_lhs_matrix_inv()
    b = step.get_rhs_const_vec()

    DinvA = Dinv @ A
    Dinvb = Dinv @ b
    lower_u = curr.iterate_vars[u].get_lower_bound()
    upper_u = curr.iterate_vars[u].get_upper_bound()
    # print(lower_u, upper_u)
    lower_y, upper_y = lin_bound_map(lower_u, upper_u, DinvA)
    # print(lower_y + b, upper_y + b)
    curr.iterate_vars[y].set_lower_bound(lower_y + Dinvb)
    curr.iterate_vars[y].set_upper_bound(upper_y + Dinvb)


def lin_bound_map(l, u, A):
    A = A.toarray()
    (m, n) = A.shape
    l_out = np.zeros(m)
    u_out = np.zeros(m)
    for i in range(m):
        lower = 0
        upper = 0
        for j in range(n):
            if A[i][j] >= 0:
                lower += A[i][j] * l[j]
                upper += A[i][j] * u[j]
            else:
                lower += A[i][j] * u[j]
                upper += A[i][j] * l[j]
        l_out[i] = lower
        u_out[i] = upper

    return np.reshape(l_out, (m, 1)), np.reshape(u_out, (m, 1))
