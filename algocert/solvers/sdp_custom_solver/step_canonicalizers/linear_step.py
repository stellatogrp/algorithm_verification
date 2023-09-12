from algocert.solvers.sdp_custom_solver.equality_constraints import (
    equality1D_constraints,
    equality2D_constraints,
)


def linear_step_canon(step, k, handler):

    D = step.get_lhs_matrix()
    A = step.get_rhs_matrix()
    step.get_rhs_matrix_blocks()
    b = step.get_rhs_const_vec()
    y = step.get_output_var()
    u = step.get_input_var()  # remember this is a stack of variables

    A_matrices, b_lvals, b_uvals = equality1D_constraints(D, y, A, u, b, k, handler)

    A2D, bl2D, bu2D = equality2D_constraints(D, y, A, u, b, k, handler)

    A_matrices += A2D
    b_lvals += bl2D
    b_uvals += bu2D

    # print(len(A_matrices), len(b_lvals), len(b_uvals))

    # exit(0)
    return A_matrices, b_lvals, b_uvals
