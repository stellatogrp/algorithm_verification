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
    xxTrange_handler = RangeHandler2D(xbounds, xbounds)

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

    if handler.add_planet:
        print('planet')
        x_upper = handler.var_upperbounds[xrange1D_handler.index_matrix()]
        x_lower = handler.var_lowerbounds[xrange1D_handler.index_matrix()]
        y_upper = handler.var_upperbounds[yrange1D_handler.index_matrix()]
        y_lower = handler.var_lowerbounds[yrange1D_handler.index_matrix()]
        # print(x_lower, x_upper)
        gaps_vec = (x_upper - x_lower).reshape(-1,)
        pos_gap_indices = np.argwhere(gaps_vec >= 1e-6).reshape(-1, )
        zero_gap_indices = np.argwhere(gaps_vec < 1e-6).reshape(-1, )
        np.argwhere(gaps_vec < 1e-5).reshape(-1, )
        print(pos_gap_indices, zero_gap_indices, gaps_vec)
        frac = np.divide((y_upper - y_lower)[pos_gap_indices], (x_upper - x_lower)[pos_gap_indices])
        # print(frac)

        # A = np.zeros((n_pos, n_pos))
        #             # print(A)
        # for i in range(n_pos):
        #     A[i, i] = frac[i, 0]
        # b = np.multiply(frac, -lower_x[pos_gap_indices]) + lower_y[pos_gap_indices]

        D = np.zeros((y_dim, y_dim))
        I = np.eye(y_dim)
        for j, i in enumerate(pos_gap_indices):
            D[i, i] = frac[j]
        c = np.multiply(frac, -x_lower[pos_gap_indices]) + y_lower[pos_gap_indices]
        minusc_xupperT = -c @ x_upper.T
        for pos_idx, i in enumerate(pos_gap_indices):
            outmat = np.zeros((problem_dim, problem_dim))
            # Di = D[i].T.reshape((-1, 1))
            # DTj = D.T[:, j].T.reshape((1, -1))
            # # print(Di.shape, DTj.shape)
            # Ai = A[i].T.reshape((-1, 1))
            # ATj = A.T[:, j].T.reshape((1, -1))
            # outmat[yrange2D_handler.index_matrix()] = DiDTj
            # outmat[urange2D_handler.index_matrix()] = -AiATj
            # outmat[urange1D_handler.index_matrix()] = -Ai * b[j, 0]
            # # print(urange1D_handler.index_matrix_horiz())
            # outmat[urange1D_handler.index_matrix_horiz()] = -b[i, 0] * ATj

            Di = D[i].T.reshape((-1, 1))
            Ii = I[i].T.reshape((-1, 1))
            ITj = I.T[:, j].T.reshape((1, -1))
            outmat[xrange1D_handler.index_matrix()] = Di * x_upper[i, 0]
            outmat[xxTrange_handler.index_matrix()] = -Di @ ITj
            outmat[xrange1D_handler.index_matrix_horiz()] = -c[pos_idx, 0] * ITj
            outmat[yrange1D_handler.index_matrix()] = -Ii * x_upper[i, 0]
            outmat[yxTrange_handler.index_matrix()] = Ii @ ITj
            outmat = (outmat + outmat.T) / 2

            A_vals.append(spa.csc_matrix(outmat))
            b_lvals.append(minusc_xupperT[pos_idx, i])
            b_uvals.append(np.inf)

        # exit(0)

    return A_vals, b_lvals, b_uvals


def nonneg_orthant_proj_bound_canon(step, k, handler):
    # print('nonneg bound')
    y = step.get_output_var()
    x = step.get_input_var()
    n = y.get_dim()

    # NOTE: assumes x update happens before proj
    yrange = handler.iter_bound_map[y][k]
    xrange = handler.iter_bound_map[x][k]

    yrange_handler = RangeHandler1D(yrange)
    xrange_handler = RangeHandler1D(xrange)

    x_lower = handler.var_lowerbounds[xrange_handler.index_matrix()]
    x_upper = handler.var_upperbounds[xrange_handler.index_matrix()]
    zeros = np.zeros((n, 1))
    y_lower = np.maximum(x_lower, zeros)
    y_upper = np.maximum(x_upper, zeros)
    handler.var_lowerbounds[yrange_handler.index_matrix()] = y_lower
    handler.var_upperbounds[yrange_handler.index_matrix()] = y_upper
