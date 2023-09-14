import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D, RangeHandler2D


def nonneg_orthant_proj_canon(step, k, handler):

    y = step.get_output_var()
    x = step.get_input_var()
    y_dim = y.get_dim()
    problem_dim = handler.problem_dim
    iter_bound_map = handler.iter_bound_map

    A_vals = []
    b_lvals = []
    b_uvals = []

    # NOTE assums that y^{k+1} = (x^{k+1})_+ (i.e. that proj does not happen first in alg)

    ybounds = iter_bound_map[y][k]
    xbounds = iter_bound_map[x][k]
    # print(ybounds, xbounds)

    yrange1D_handler = RangeHandler1D(ybounds)
    xrange1D_handler = RangeHandler1D(xbounds)
    yyTrange_handler = RangeHandler2D(ybounds, ybounds)
    yxTrange_handler = RangeHandler2D(ybounds, xbounds)

    # First, y >= 0 TODO after looking at RLT, see if this can be removed
    for i in range(y_dim):
        output_mat = np.zeros((problem_dim, problem_dim))
        insert_vec = np.zeros((y_dim, 1))
        insert_vec[i, 0] = 1
        output_mat[yrange1D_handler.index_matrix()] = insert_vec
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(0)
        b_uvals.append(np.inf)

    # Second x >= 0
    for i in range(y_dim):
        output_mat = np.zeros((problem_dim, problem_dim))
        insert_vec = np.zeros((y_dim, 1))
        insert_vec[i, 0] = 1
        output_mat[yrange1D_handler.index_matrix()] = insert_vec
        output_mat[xrange1D_handler.index_matrix()] = -insert_vec
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(0)
        b_uvals.append(np.inf)
        # print(output_mat)

    # Lastly diag(yyT - yxT) = 0
    for i in range(y_dim):
        output_mat = np.zeros((problem_dim, problem_dim))
        insert_mat = np.zeros((y_dim, y_dim))
        insert_mat[i, i] = 1
        output_mat[yyTrange_handler.index_matrix()] = insert_mat
        output_mat[yxTrange_handler.index_matrix()] = -insert_mat
        output_mat = (output_mat + output_mat.T) / 2
        # print(output_mat)
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(0)
        b_uvals.append(0)


    # TODO check trace again after adding in bounds/RLT
    # outmat = np.zeros((problem_dim, problem_dim))
    # output_mat[yyTrange_handler.index_matrix()] = np.eye(y_dim)
    # output_mat[yxTrange_handler.index_matrix()] = -np.eye(y_dim)
    # output_mat = (output_mat + output_mat.T) / 2
    # A_vals.append(spa.csc_matrix(output_mat))
    # b_lvals.append(0)
    # b_uvals.append(0)

    # print(len(A_vals), len(b_lvals), len(b_uvals))
    # exit(0)

    return A_vals, b_lvals, b_uvals


def nonneg_orthant_proj_bound_canon(step, k, handler):
    print('nonneg bound')
    exit(0)
