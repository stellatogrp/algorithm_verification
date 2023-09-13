import numpy as np

from algocert.solvers.sdp_custom_solver.cross_constraints import cross_constraints_from_ranges
from algocert.solvers.sdp_custom_solver.equality_constraints import (
    equality1D_constraints,
    equality2D_constraints,
)
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
