import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D, RangeHandler2D


def RLT_from_ranges(yrange, xrange, handler, upper_triangle_only=True):
    print('add RLT')
    RangeHandler1D(yrange)
    RangeHandler1D(xrange)
    yxTrange_handler = RangeHandler2D(yrange, xrange)
    m, n = yxTrange_handler.shape

    A_vals = []
    b_lvals = []
    b_uvals = []

    for i in range(m):
        if upper_triangle_only:
            jrange = range(i, n)
        else:
            jrange = range(n)
        for j in jrange:
            print(i, j)

    exit(0)
    return A_vals, b_lvals, b_uvals


def RLT_diagonal_vars(matrix_dim, handler):
    A_vals = []
    b_lvals = []
    b_uvals = []

    u = handler.var_upperbounds

    for i in range(matrix_dim):
        j = i
        output_mat4 = np.zeros((matrix_dim + 1, matrix_dim + 1))
        output_mat4[i, j] = 1
        output_mat4[j, -1] = -u[i, 0]
        output_mat4[-1, i] = -u[j, 0]
        output_mat4 = (output_mat4 + output_mat4.T) / 2
        A_vals.append(spa.csc_matrix(output_mat4))
        b_lvals.append(-u[i, 0] * u[j, 0])
        b_uvals.append(np.inf)

    return A_vals, b_lvals, b_uvals


def RLT_all_vars(matrix_dim, handler):
    print('adding mat RLT')
    A_vals = []
    b_lvals = []
    b_uvals = []

    l = handler.var_lowerbounds
    u = handler.var_upperbounds

    for i in range(matrix_dim):
        for j in range(i, matrix_dim):
            # # output_mat1 = np.zeros((matrix_dim + 1, matrix_dim + 1))
            # output_mat1 = spa.lil_matrix((matrix_dim + 1, matrix_dim + 1))
            # output_mat1[i, j] = 1
            # output_mat1[j, -1] = -l[i, 0]
            # output_mat1[-1, i] = -l[j, 0]
            # output_mat1 = (output_mat1 + output_mat1.T) / 2
            # A_vals.append(spa.csc_matrix(output_mat1))
            # b_lvals.append(-l[i, 0] * l[j, 0])
            # b_uvals.append(np.inf)

            # # output_mat2 = np.zeros((matrix_dim + 1, matrix_dim + 1))
            # output_mat2 = spa.lil_matrix((matrix_dim + 1, matrix_dim + 1))
            # output_mat2[i, j] = -1
            # output_mat2[j, -1] = l[i, 0]
            # output_mat2[-1, i] = u[j, 0]
            # output_mat2 = (output_mat2 + output_mat2.T) / 2
            # A_vals.append(spa.csc_matrix(output_mat2))
            # b_lvals.append(l[i, 0] * u[j, 0])
            # b_uvals.append(np.inf)

            # output_mat3 = np.zeros((matrix_dim + 1, matrix_dim + 1))
            output_mat3 = spa.lil_matrix((matrix_dim + 1, matrix_dim + 1))
            output_mat3[i, j] = -1
            output_mat3[j, -1] = u[i, 0]
            output_mat3[-1, i] = l[j, 0]
            output_mat3 = (output_mat3 + output_mat3.T) / 2
            A_vals.append(spa.csc_matrix(output_mat3))
            b_lvals.append(u[i, 0] * l[j, 0])
            b_uvals.append(np.inf)

            # output_mat4 = np.zeros((matrix_dim + 1, matrix_dim + 1))
            output_mat4 = spa.lil_matrix((matrix_dim + 1, matrix_dim + 1))
            output_mat4[i, j] = 1
            output_mat4[j, -1] = -u[i, 0]
            output_mat4[-1, i] = -u[j, 0]
            output_mat4 = (output_mat4 + output_mat4.T) / 2
            A_vals.append(spa.csc_matrix(output_mat4))
            b_lvals.append(-u[i, 0] * u[j, 0])
            b_uvals.append(np.inf)

    # exit(0)
    return A_vals, b_lvals, b_uvals
