import numpy as np

from algocert.solvers.sdp_custom_solver.cross_constraints import cross_constraints_from_ranges
from algocert.solvers.sdp_custom_solver.equality_constraints import (
    equality1D_constraints,
    equality2D_constraints,
)
from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D
from algocert.solvers.sdp_custom_solver.utils import map_linstep_to_ranges


def linear_step_canon(step, k, handler):

    D = step.get_lhs_matrix()
    A = step.get_rhs_matrix()
    step.get_rhs_matrix_blocks()
    b = step.get_rhs_const_vec()
    y = step.get_output_var()
    u = step.get_input_var()  # remember this is a stack of variables
    u_dim = step.get_input_var_dim()

    A_matrices, b_lvals, b_uvals = equality1D_constraints(D, y, A, u, b, k, handler)

    A2D, bl2D, bu2D = equality2D_constraints(D, y, A, u, b, k, handler)

    A_matrices += A2D
    b_lvals += bl2D
    b_uvals += bu2D

    iter_bound_map = handler.iter_bound_map
    yrange = iter_bound_map[y][k]
    urange = map_linstep_to_ranges(y, u, k, handler)

    C = np.eye(u_dim)
    Across, blcross, bucross = cross_constraints_from_ranges(y.get_dim(), u_dim, handler.problem_dim,
                                                             D.todense(), yrange, A.todense(), urange, b,
                                                             C, urange, C, urange, np.zeros((u_dim, 1)))

    A_matrices += Across
    b_lvals += blcross
    b_uvals += bucross

    # print(len(A_matrices), len(b_lvals), len(b_uvals))

    # exit(0)
    return A_matrices, b_lvals, b_uvals


def linear_step_bound_canon(step, k, handler):
    # print('lin step bound')
    # D = step.get_lhs_matrix()
    u = step.get_input_var()  # remember this is a list of vars
    y = step.get_output_var()
    A = step.get_rhs_matrix()
    b = step.get_rhs_const_vec()

    DinvA = step.solve_linear_system(A.todense())
    Dinvb = step.solve_linear_system(b)

    yrange = handler.iter_bound_map[y][k]
    uranges = map_linstep_to_ranges(y, u, k, handler)

    # print(yrange, uranges)

    yrange_handler = RangeHandler1D(yrange)
    urange_handler = RangeHandler1D(uranges)
    u_lower = handler.var_lowerbounds[urange_handler.index_matrix()]
    u_upper = handler.var_upperbounds[urange_handler.index_matrix()]
    # print(u_lower, u_upper)

    y_lower, y_upper = lin_bound_map(u_lower, u_upper, DinvA)
    handler.var_lowerbounds[yrange_handler.index_matrix()] = y_lower + Dinvb
    handler.var_upperbounds[yrange_handler.index_matrix()] = y_upper + Dinvb


def lin_bound_map(l, u, A):
    # A = A.toarray()
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
