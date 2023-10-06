import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.psd_cone_handler import PSDConeHandler
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
    psd_cone_handlers = []

    # NOTE assums that y^{k+1} = (x^{k+1})_+ (i.e. that proj does not happen first in alg)

    ybounds = iter_bound_map[y][k]
    xbounds = iter_bound_map[x][k]
    # print(ybounds, xbounds)

    yrange1D_handler = RangeHandler1D(ybounds)
    xrange1D_handler = RangeHandler1D(xbounds)
    yyTrange_handler = RangeHandler2D(ybounds, ybounds)
    yxTrange_handler = RangeHandler2D(ybounds, xbounds)
    xxTrange_handler = RangeHandler2D(xbounds, xbounds)

    psd_cone_handlers.append(PSDConeHandler([ybounds, xbounds]))
    # print(ybounds, xbounds, PSDConeHandler([ybounds, xbounds]).ranges_dim)
    # exit(0)
    real_indices = np.array(step.real_indices)
    nonneg_indices = np.array(step.nonneg_indices)
    # nonneg_indices = np.array(list(range(y_dim)))

    # print(real_indices, nonneg_indices)
    # exit(0)

    # TODO rewrite by adding equality constraints between real indices

    # First, y >= 0 if RLT is not already added
    if not handler.add_RLT:
        for i in range(y_dim):
            if i not in nonneg_indices:
                continue
            # output_mat = np.zeros((problem_dim, problem_dim))
            output_mat = spa.lil_matrix((problem_dim, problem_dim))
            insert_vec = np.zeros((y_dim, 1))
            insert_vec[i, 0] = 1
            output_mat[yrange1D_handler.index_matrix()] = insert_vec
            output_mat = (output_mat + output_mat.T) / 2
            A_vals.append(spa.csc_matrix(output_mat))
            b_lvals.append(0)
            b_uvals.append(np.inf)

    # Second y >= x or y == x depending on if i should be nonneg or not
    for i in range(y_dim):
        if i not in nonneg_indices:
            continue
        # output_mat = np.zeros((problem_dim, problem_dim))
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_vec = np.zeros((y_dim, 1))
        insert_vec[i, 0] = 1
        output_mat[yrange1D_handler.index_matrix()] = insert_vec
        output_mat[xrange1D_handler.index_matrix()] = -insert_vec
        output_mat = (output_mat + output_mat.T) / 2
        # print(output_mat, spa.csc_matrix(output_mat))
        # exit(0)
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(0)
        # if i in nonneg_indices:
        #     b_uvals.append(np.inf)
        # else:
        #     b_uvals.append(0)
        b_uvals.append(np.inf)
        # print(output_mat)

    # Lastly diag(yyT - yxT) = 0
    for i in range(y_dim):
        if i not in nonneg_indices:
            continue
        # output_mat = np.zeros((problem_dim, problem_dim))
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_mat = np.zeros((y_dim, y_dim))
        insert_mat[i, i] = 1
        output_mat[yyTrange_handler.index_matrix()] = insert_mat
        # if i in nonneg_indices:
        #     output_mat[yxTrange_handler.index_matrix()] = -insert_mat
        # else:
        #     output_mat[xxTrange_handler.index_matrix()] = -insert_mat
        output_mat[yxTrange_handler.index_matrix()] = -insert_mat
        output_mat = (output_mat + output_mat.T) / 2
        # print(output_mat)
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(0)
        b_uvals.append(0)

    # yxT_var @ D.T == yuT_var @ A.T + y_var @ b.T,
    # if x in handler.linstep_output_vars:
    #     xstep = handler.var_linstep_map[x]
    #     D = xstep.get_lhs_matrix()
    #     A = xstep.get_rhs_matrix()
    #     b = xstep.get_rhs_const_vec()
    #     u = xstep.get_input_var()
    #     ubounds = map_linstep_to_ranges(x, u, k, handler)
    #     # print(uranges)
    #     C = np.eye(y.get_dim())
    #     # A_cross, b_lcross, b_ucross = cross_constraints_linstep_to_not(y.get_dim(), x.get_dim(), problem_dim,
    #     #                                                                C, ybounds, C, ybounds, np.zeros((y.get_dim(), 1)),
    #     #                                                                D, xbounds, A, ubounds, b)
    #     A_cross, b_lcross, b_ucross = cross_constraints_from_ranges(y.get_dim(), x.get_dim(), problem_dim,
    #                                                                 C, ybounds, C, ybounds, np.zeros((y.get_dim(), 1)),
    #                                                                 D, xbounds, A, ubounds, b)
    #     A_vals += A_cross
    #     b_lvals += b_lcross
    #     b_uvals += b_ucross
    # #     # exit(0)

    # TODO check trace again after adding in bounds/RLT
    # outmat = np.zeros((problem_dim, problem_dim))
    # output_mat = spa.lil_matrix((problem_dim, problem_dim))
    # output_mat[yyTrange_handler.index_matrix()] = np.eye(y_dim)
    # output_mat[yxTrange_handler.index_matrix()] = -np.eye(y_dim)
    # output_mat = (output_mat + output_mat.T) / 2
    # A_vals.append(spa.csc_matrix(output_mat))
    # b_lvals.append(0)
    # b_uvals.append(0)

    # print(len(A_vals), len(b_lvals), len(b_uvals))
    # exit(0)

    if handler.add_planet:
        # print('planet')
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

        D = np.zeros((y_dim, y_dim))
        I = np.eye(y_dim)
        for j, i in enumerate(pos_gap_indices):
            D[i, i] = frac[j]
        c = np.multiply(frac, -x_lower[pos_gap_indices]) + y_lower[pos_gap_indices]
        minusc_xupperT = -c @ x_upper.T
        for pos_idx, i in enumerate(pos_gap_indices):
            if i not in nonneg_indices:
                continue
            # print(pos_idx, i, j)
            # j = pos_idx
            # if i not in nonneg_indices:
            #     continue
            # outmat = np.zeros((problem_dim, problem_dim))
            outmat = spa.lil_matrix((problem_dim, problem_dim))

            Di = D[i].T.reshape((-1, 1))
            Ii = I[i].T.reshape((-1, 1))
            ITj = I.T[:, i].T.reshape((1, -1))
            outmat[xrange1D_handler.index_matrix()] = Di * x_upper[i, 0]
            outmat[xxTrange_handler.index_matrix()] = -Di @ ITj
            outmat[xrange1D_handler.index_matrix_horiz()] = -c[pos_idx, 0] * ITj
            outmat[yrange1D_handler.index_matrix()] = -Ii * x_upper[i, 0]
            outmat[yxTrange_handler.index_matrix()] = Ii @ ITj
            outmat = (outmat + outmat.T) / 2

            A_vals.append(spa.csc_matrix(outmat))
            b_lvals.append(minusc_xupperT[pos_idx, i])
            b_uvals.append(np.inf)

    real_A, real_bl, real_bu = handle_real_indices(y_dim, ybounds, xbounds, real_indices, problem_dim)
    A_vals += real_A
    b_lvals += real_bl
    b_uvals += real_bu

    # print(len(A_vals), len(b_lvals), len(b_uvals))
    # exit(0)
    return A_vals, b_lvals, b_uvals, psd_cone_handlers


def handle_real_indices(n, ybounds, xbounds, real_indices, problem_dim):
    A_vals = []
    b_lvals = []
    b_uvals = []
    yyTrange_handler = RangeHandler2D(ybounds, ybounds)
    yxTrange_handler = RangeHandler2D(ybounds, xbounds)
    xxTrange_handler = RangeHandler2D(xbounds, xbounds)
    for i in real_indices:
        # for j in real_indices:
        j = i
        insert_mat = np.zeros((n, n))
        insert_mat[i, j] = 1
        # print(i, j)

        # first yiyj - xixj = 0
        # output_mat1 = np.zeros((problem_dim, problem_dim))
        output_mat1 = spa.lil_matrix((problem_dim, problem_dim))
        output_mat1[yyTrange_handler.index_matrix()] = insert_mat
        output_mat1[xxTrange_handler.index_matrix()] = -insert_mat
        output_mat1 = (output_mat1 + output_mat1.T) / 2

        A_vals.append(spa.csc_matrix(output_mat1))
        b_lvals.append(0)
        b_uvals.append(0)

        # second yiyj - yixj = 0
        # output_mat2 = np.zeros((problem_dim, problem_dim))
        output_mat2 = spa.lil_matrix((problem_dim, problem_dim))
        output_mat2[yyTrange_handler.index_matrix()] = insert_mat
        output_mat2[yxTrange_handler.index_matrix()] = -insert_mat
        output_mat2 = (output_mat2 + output_mat2.T) / 2

        A_vals.append(spa.csc_matrix(output_mat2))
        b_lvals.append(0)
        b_uvals.append(0)

        # third yiyj - yjxi = 0 (maybe?) TODO see if needed
        # output_mat3 = np.zeros((problem_dim, problem_dim))
        # output_mat = spa.lil_matrix((problem_dim, problem_dim))
        # output_mat3[yyTrange_handler.index_matrix()] = insert_mat
        # output_mat3[yxTrange_handler.index_matrix()] = -insert_mat.T
        # output_mat3 = (output_mat3 + output_mat3.T) / 2

        # A_vals.append(spa.csc_matrix(output_mat3))
        # b_lvals.append(0)
            # b_uvals.append(0)

    return A_vals, b_lvals, b_uvals

def nonneg_orthant_proj_bound_canon(step, k, handler):
    # print('nonneg bound')
    y = step.get_output_var()
    x = step.get_input_var()
    # y.get_dim()
    nonneg_indices = step.nonneg_indices

    # NOTE: assumes x update happens before proj
    yrange = handler.iter_bound_map[y][k]
    xrange = handler.iter_bound_map[x][k]

    yrange_handler = RangeHandler1D(yrange)
    xrange_handler = RangeHandler1D(xrange)

    x_lower = handler.var_lowerbounds[xrange_handler.index_matrix()]
    x_upper = handler.var_upperbounds[xrange_handler.index_matrix()]

    zeros = np.zeros((len(nonneg_indices), 1))
    y_lower = x_lower.copy()
    y_upper = x_upper.copy()
    y_lower[nonneg_indices] = np.maximum(y_lower[nonneg_indices], zeros)
    y_upper[nonneg_indices] = np.maximum(y_upper[nonneg_indices], zeros)

    # zeros = np.zeros((n, 1))
    # y_lower = np.maximum(x_lower, zeros)
    # y_upper = np.maximum(x_upper, zeros)

    handler.var_lowerbounds[yrange_handler.index_matrix()] = y_lower
    handler.var_upperbounds[yrange_handler.index_matrix()] = y_upper
    # print(x_lower, x_upper, y_lower, y_upper)
    # exit(0)
