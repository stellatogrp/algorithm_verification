import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.psd_cone_handler import PSDConeHandler
from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D, RangeHandler2D
from algocert.variables.parameter import Parameter


def max_with_vec_step_canon(step, k, handler):
    y = step.get_output_var()
    n = y.get_dim()
    x = step.get_input_var()
    l = step.get_lower_bound_vec()
    problem_dim = handler.problem_dim

    if isinstance(l, Parameter):
        A_vals, b_lvals, b_uvals, psd_cones_handlers = canon_with_l_param(step, k, handler)
    else:
        A_vals, b_lvals, b_uvals, psd_cones_handlers = canon_with_l_const(step, k, handler)

    if handler.add_planet:
        ybounds = handler.iter_bound_map[y][k]
        xbounds = handler.iter_bound_map[x][k]
        yrange1D_handler = RangeHandler1D(ybounds)
        xrange1D_handler = RangeHandler1D(xbounds)
        RangeHandler2D(ybounds, ybounds)
        yxTrange_handler = RangeHandler2D(ybounds, xbounds)
        xxTrange_handler = RangeHandler2D(xbounds, xbounds)

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

        D = np.zeros((n, n))
        I = np.eye(n)
        for j, i in enumerate(pos_gap_indices):
            D[i, i] = frac[j]
        c = np.multiply(frac, -x_lower[pos_gap_indices]) + y_lower[pos_gap_indices]
        minusc_xupperT = -c @ x_upper.T
        for pos_idx, i in enumerate(pos_gap_indices):
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

    return A_vals, b_lvals, b_uvals, psd_cones_handlers


def canon_with_l_param(step, k, handler):
    y = step.get_output_var()
    x = step.get_input_var()
    l = step.get_lower_bound_vec()
    n = y.get_dim()
    problem_dim = handler.problem_dim
    iter_bound_map = handler.iter_bound_map

    # NOTE assumes that y^{k+1} = max(x^{k+1}, l) (i.e. that proj does not happen first in alg)

    A_vals = []
    b_lvals = []
    b_uvals = []
    psd_cone_handlers = []

    ybounds = iter_bound_map[y][k]
    xbounds = iter_bound_map[x][k]
    lbounds = handler.param_bound_map[l]
    lrange1D_handler = RangeHandler1D(lbounds)
    psd_cone_handlers.append(PSDConeHandler([ybounds, xbounds, lbounds]))

    yrange1D_handler = RangeHandler1D(ybounds)
    xrange1D_handler = RangeHandler1D(xbounds)
    yyTrange_handler = RangeHandler2D(ybounds, ybounds)
    yxTrange_handler = RangeHandler2D(ybounds, xbounds)
    RangeHandler2D(xbounds, xbounds)

    lyTrange_handler = RangeHandler2D(lbounds, ybounds)
    lxTrange_handler = RangeHandler2D(lbounds, xbounds)

    # y - l >= 0
    for i in range(n):
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_vec = np.zeros((n, 1))
        insert_vec[i, 0] = 1
        output_mat[yrange1D_handler.index_matrix()] = insert_vec
        output_mat[lrange1D_handler.index_matrix()] = -insert_vec
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(0)
        b_uvals.append(np.inf)

    # y - x >= 0
    for i in range(n):
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_vec = np.zeros((n, 1))
        insert_vec[i, 0] = 1
        output_mat[yrange1D_handler.index_matrix()] = insert_vec
        output_mat[xrange1D_handler.index_matrix()] = -insert_vec
        output_mat = (output_mat + output_mat.T) / 2
        # print(output_mat, spa.csc_matrix(output_mat))
        # exit(0)
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(0)
        b_uvals.append(np.inf)

    # diag(yyT - yxT - lyT + lxT) == 0
    for i in range(n):
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_mat = np.zeros((n, n))
        insert_mat[i, i] = 1
        output_mat[yyTrange_handler.index_matrix()] = insert_mat
        output_mat[yxTrange_handler.index_matrix()] = -insert_mat
        output_mat[lyTrange_handler.index_matrix()] = -insert_mat
        output_mat[lxTrange_handler.index_matrix()] = insert_mat
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(0)
        b_uvals.append(0)

    return A_vals, b_lvals, b_uvals, psd_cone_handlers


def canon_with_l_const(step, k, handler):
    y = step.get_output_var()
    x = step.get_input_var()
    l = step.get_lower_bound_vec()
    n = y.get_dim()
    problem_dim = handler.problem_dim
    iter_bound_map = handler.iter_bound_map

    A_vals = []
    b_lvals = []
    b_uvals = []
    psd_cone_handlers = []

    ybounds = iter_bound_map[y][k]
    xbounds = iter_bound_map[x][k]
    psd_cone_handlers.append(PSDConeHandler([ybounds, xbounds]))

    yrange1D_handler = RangeHandler1D(ybounds)
    xrange1D_handler = RangeHandler1D(xbounds)
    yyTrange_handler = RangeHandler2D(ybounds, ybounds)
    yxTrange_handler = RangeHandler2D(ybounds, xbounds)

    # y >= l
    if not handler.add_RLT:
        for i in range(n):
            output_mat = spa.lil_matrix((problem_dim, problem_dim))
            insert_vec = np.zeros((n, 1))
            insert_vec[i, 0] = 1
            output_mat[yrange1D_handler.index_matrix()] = insert_vec
            # output_mat[urange1D_handler.index_matrix()] = -insert_vec
            output_mat = (output_mat + output_mat.T) / 2
            A_vals.append(spa.csc_matrix(output_mat))
            b_lvals.append(l[i, 0])
            b_uvals.append(np.inf)

    # y - x >= 0
    for i in range(n):
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_vec = np.zeros((n, 1))
        insert_vec[i, 0] = 1
        output_mat[yrange1D_handler.index_matrix()] = insert_vec
        output_mat[xrange1D_handler.index_matrix()] = -insert_vec
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(0)
        b_uvals.append(np.inf)

    # diag(yyT - yxT - lyT + lxT) == 0
    for i in range(n):
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_mat = np.zeros((n, n))
        insert_vec = np.zeros((n, 1))
        insert_mat[i, i] = 1
        insert_vec[i, 0] = l[i, 0]
        output_mat[yyTrange_handler.index_matrix()] = insert_mat
        output_mat[yxTrange_handler.index_matrix()] = -insert_mat
        # output_mat[lyTrange_handler.index_matrix()] = -insert_mat
        # output_mat[lxTrange_handler.index_matrix()] = insert_mat
        output_mat[yrange1D_handler.index_matrix()] = -insert_vec
        output_mat[xrange1D_handler.index_matrix()] = insert_vec
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(0)
        b_uvals.append(0)

    return A_vals, b_lvals, b_uvals, psd_cone_handlers


def max_with_vec_bound_canon(step, k, handler):
    y = step.get_output_var()
    x = step.get_input_var()
    l = step.get_lower_bound_vec()

    # NOTE: assumes x update happens before proj
    yrange = handler.iter_bound_map[y][k]
    xrange = handler.iter_bound_map[x][k]

    yrange_handler = RangeHandler1D(yrange)
    xrange_handler = RangeHandler1D(xrange)

    if not isinstance(l, Parameter):
        l_vec = l.reshape((-1, 1))
        l_lower = l_vec
        l_upper = l_vec
    else:
        lrange = handler.param_bound_map[l]
        lrange_handler = RangeHandler1D(lrange)
        l_lower = handler.var_lowerbounds[lrange_handler.index_matrix()]
        l_upper = handler.var_upperbounds[lrange_handler.index_matrix()]

    x_lower = handler.var_lowerbounds[xrange_handler.index_matrix()]
    x_upper = handler.var_upperbounds[xrange_handler.index_matrix()]

    # y_lower = np.maximum(x_lower, zeros)
    # y_upper = np.maximum(x_upper, zeros)

    y_lower = np.maximum(x_lower, l_lower)
    y_upper = np.maximum(x_upper, l_upper)

    # print(x_lower, l_lower, y_lower)
    # print(x_upper, l_upper, y_upper)

    handler.var_lowerbounds[yrange_handler.index_matrix()] = y_lower
    handler.var_upperbounds[yrange_handler.index_matrix()] = y_upper
