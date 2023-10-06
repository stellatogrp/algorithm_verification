import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.psd_cone_handler import PSDConeHandler
from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D, RangeHandler2D
from algocert.variables.parameter import Parameter


def min_with_vec_step_canon(step, k, handler):
    y = step.get_output_var()
    n = y.get_dim()
    x = step.get_input_var()
    u = step.get_upper_bound_vec()
    problem_dim = handler.problem_dim

    if isinstance(u, Parameter):
        A_vals, b_lvals, b_uvals, psd_cones_handlers = canon_with_u_param(step, k, handler)
    else:
        A_vals, b_lvals, b_uvals, psd_cones_handlers = canon_with_u_const(step, k, handler)

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
        minusc_xlowerT = -c @ x_lower.T
        for pos_idx, i in enumerate(pos_gap_indices):
            outmat = spa.lil_matrix((problem_dim, problem_dim))

            Di = D[i].T.reshape((-1, 1))
            Ii = I[i].T.reshape((-1, 1))
            ITj = I.T[:, i].T.reshape((1, -1))
            outmat[xrange1D_handler.index_matrix()] = Di * x_lower[i, 0]
            outmat[xxTrange_handler.index_matrix()] = -Di @ ITj
            outmat[xrange1D_handler.index_matrix_horiz()] = -c[pos_idx, 0] * ITj
            outmat[yrange1D_handler.index_matrix()] = -Ii * x_lower[i, 0]
            outmat[yxTrange_handler.index_matrix()] = Ii @ ITj
            outmat = (outmat + outmat.T) / 2

            A_vals.append(spa.csc_matrix(outmat))
            b_lvals.append(minusc_xlowerT[pos_idx, i])
            b_uvals.append(np.inf)

    return A_vals, b_lvals, b_uvals, psd_cones_handlers


def canon_with_u_param(step, k, handler):
    y = step.get_output_var()
    x = step.get_input_var()
    u = step.get_upper_bound_vec()
    n = y.get_dim()
    problem_dim = handler.problem_dim
    iter_bound_map = handler.iter_bound_map

    # NOTE assumes that y^{k+1} = min(x^{k+1}, u) (i.e. that proj does not happen first in alg)

    A_vals = []
    b_lvals = []
    b_uvals = []
    psd_cone_handlers = []

    ybounds = iter_bound_map[y][k]
    xbounds = iter_bound_map[x][k]
    ubounds = handler.param_bound_map[u]
    urange1D_handler = RangeHandler1D(ubounds)
    psd_cone_handlers.append(PSDConeHandler([ybounds, xbounds, ubounds]))

    yrange1D_handler = RangeHandler1D(ybounds)
    xrange1D_handler = RangeHandler1D(xbounds)
    yyTrange_handler = RangeHandler2D(ybounds, ybounds)
    yxTrange_handler = RangeHandler2D(ybounds, xbounds)
    RangeHandler2D(xbounds, xbounds)

    uyTrange_handler = RangeHandler2D(ubounds, ybounds)
    uxTrange_handler = RangeHandler2D(ubounds, xbounds)

    # y - u <= 0
    for i in range(n):
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_vec = np.zeros((n, 1))
        insert_vec[i, 0] = 1
        output_mat[yrange1D_handler.index_matrix()] = insert_vec
        output_mat[urange1D_handler.index_matrix()] = -insert_vec
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(-np.inf)
        b_uvals.append(0)

    # y - x <= 0
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
        b_lvals.append(-np.inf)
        b_uvals.append(0)

    # diag(yyT - yxT - uyT + uxT) == 0
    for i in range(n):
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_mat = np.zeros((n, n))
        insert_mat[i, i] = 1
        output_mat[yyTrange_handler.index_matrix()] = insert_mat
        output_mat[yxTrange_handler.index_matrix()] = -insert_mat
        output_mat[uyTrange_handler.index_matrix()] = -insert_mat
        output_mat[uxTrange_handler.index_matrix()] = insert_mat
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(0)
        b_uvals.append(0)

    return A_vals, b_lvals, b_uvals, psd_cone_handlers


def canon_with_u_const(step, k, handler):
    y = step.get_output_var()
    x = step.get_input_var()
    u = step.get_upper_bound_vec()
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

    # y <= u
    for i in range(n):
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_vec = np.zeros((n, 1))
        insert_vec[i, 0] = 1
        output_mat[yrange1D_handler.index_matrix()] = insert_vec
        # output_mat[urange1D_handler.index_matrix()] = -insert_vec
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(-np.inf)
        b_uvals.append(u[i, 0])

    # y - x <= 0
    for i in range(n):
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_vec = np.zeros((n, 1))
        insert_vec[i, 0] = 1
        output_mat[yrange1D_handler.index_matrix()] = insert_vec
        output_mat[xrange1D_handler.index_matrix()] = -insert_vec
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(-np.inf)
        b_uvals.append(0)

    # diag(yyT - yxT - uyT + uxT) == 0
    for i in range(n):
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_mat = np.zeros((n, n))
        insert_vec = np.zeros((n, 1))
        insert_mat[i, i] = 1
        insert_vec[i, 0] = u[i, 0]
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


def min_with_vec_bound_canon(step, k, handler):
    y = step.get_output_var()
    x = step.get_input_var()
    u = step.get_upper_bound_vec()

    # NOTE: assumes x update happens before proj
    yrange = handler.iter_bound_map[y][k]
    xrange = handler.iter_bound_map[x][k]

    yrange_handler = RangeHandler1D(yrange)
    xrange_handler = RangeHandler1D(xrange)

    if not isinstance(u, Parameter):
        u_vec = u.reshape((-1, 1))
        u_lower = u_vec
        u_upper = u_vec
    else:
        urange = handler.param_bound_map[u]
        urange_handler = RangeHandler1D(urange)
        u_lower = handler.var_lowerbounds[urange_handler.index_matrix()]
        u_upper = handler.var_upperbounds[urange_handler.index_matrix()]

    x_lower = handler.var_lowerbounds[xrange_handler.index_matrix()]
    x_upper = handler.var_upperbounds[xrange_handler.index_matrix()]

    # y_lower = np.minimum(x_lower, zeros)
    # y_upper = np.minimum(x_upper, zeros)

    y_lower = np.minimum(x_lower, u_lower)
    y_upper = np.minimum(x_upper, u_upper)

    # print('lower:', x_lower, u_lower, y_lower)
    # print('upper:', x_upper, u_upper, y_upper)

    handler.var_lowerbounds[yrange_handler.index_matrix()] = y_lower
    handler.var_upperbounds[yrange_handler.index_matrix()] = y_upper
