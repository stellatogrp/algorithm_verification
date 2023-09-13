import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D, RangeHandler2D


def cross_constraint_steps(step1, k1, step2, k2, handler):
    # D = step.get_lhs_matrix()
    # A = step.get_rhs_matrix()
    # step.get_rhs_matrix_blocks()
    # b = step.get_rhs_const_vec()
    # y = step.get_output_var()
    # u = step.get_input_var()
    step1.get_lhs_matrix()
    step1.get_rhs_matrix()
    step1.get_rhs_const_vec()
    step1.get_output_var()
    step1.get_input_var()

    step2.get_lhs_matrix()
    step2.get_rhs_matrix()
    step2.get_rhs_const_vec()
    step2.get_output_var()
    step2.get_input_var()

    return [], [], []


def cross_constraints(D, y, A, u, b, C, z, F, x, c, k1, k2):
    print(D)
    return [], [], []


def cross_constraints_from_ranges(m, n, problem_dim,
                                  D, y_range, A, u_range, b,
                                  C, z_range, F, x_range, c):
    '''
        Assumes D, A, C, F are dense
    '''
    # print(D)
    # print(y_range)
    # print(A)
    # print(u_range)
    # print(b)

    # print(C)
    # print(z_range)
    # print(F)
    # print(x_range)
    # print(x)

    A_matrices = []
    b_lvals = []
    b_uvals = []

    bcT = b @ c.T
    # print(bcT.shape, m, n)

    yzrange_handler = RangeHandler2D(y_range, z_range)
    uxrange_handler = RangeHandler2D(u_range, x_range)
    urange_handler = RangeHandler1D(u_range)
    xrange_handler = RangeHandler1D(x_range)
    # yrange_handler = RangeHandler1D(y_range)
    # zrange_handler = RangeHandler1D(z_range)

    for i in range(m):
        for j in range(n):
            # print(i, j)
            outmat = np.zeros((problem_dim, problem_dim))
            Di = D[i].T.reshape((-1, 1))
            CTj = C.T[:, j].T.reshape((1, -1))
            # print(Di, CTj)
            Ai = A[i].T.reshape((-1, 1))
            FTj = F.T[:, j].T.reshape((1, -1))
            bi = b[i, 0]
            cj = c[j, 0]

            # print(Di.shape, CTj.shape)
            outmat[yzrange_handler.index_matrix()] = Di @ CTj
            outmat[uxrange_handler.index_matrix()] = -Ai @ FTj
            outmat[urange_handler.index_matrix()] = -Ai * cj
            outmat[xrange_handler.index_matrix_horiz()] = -bi * FTj

            outmat = (outmat + outmat.T) / 2
            outmat = spa.csc_matrix(outmat)
            A_matrices.append(outmat)
            b_lvals.append(bcT[i, j])
            b_uvals.append(bcT[i, j])

    return A_matrices, b_lvals, b_uvals
