# TODO: Check https://github.com/cvxpy/cvxpy/blob/fb1f271b94edcbfe5f02fa4bd93ef6bb9fbdd269/cvxpy/reductions/eliminate_pwl/atom_canonicalizers/abs_canon.py
import cvxpy as cp
import numpy as np


def linear_step_canon(step, curr, prev, iter_id_map):
    u = step.get_rhs_var()
    y = step.get_output_var()
    A = step.get_matrix()
    y_var = curr.iterate_vars[y]
    yyT_var = curr.iterate_outerproduct_vars[y]
    u_var = curr.iterate_vars[u]
    uuT_var = curr.iterate_outerproduct_vars[u]
    yuT_var = curr.iterate_cross_vars[y][u]
    constraints = [y_var == A @ u_var, yyT_var == A @ uuT_var @ A.T, yuT_var == A @ uuT_var]
    constraints += [
        cp.bmat([
            [yyT_var, yuT_var, y_var],
            [yuT_var.T, uuT_var, u_var],
            [y_var.T, u_var.T, np.array([[1]])]
        ]) >> 0,
    ]
    return constraints
