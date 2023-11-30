import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.cross_constraints import cross_constraints_from_ranges
from algocert.solvers.sdp_custom_solver.equality_constraints import (
    equality1D_constraints,
    equality2D_constraints,
)
from algocert.solvers.sdp_custom_solver.psd_cone_handler import PSDConeHandler
from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D, RangeHandler2D
from algocert.solvers.sdp_custom_solver.RLT_constraints import RLT_all_in_range_list
from algocert.solvers.sdp_custom_solver.step_canonicalizers.linear_step_propagation import (
    SET_LINPROP_MAP,
)
from algocert.solvers.sdp_custom_solver.utils import map_linstep_to_iters, map_linstep_to_ranges
from algocert.variables.parameter import Parameter


def linear_max_proj_canon(step, k, handler):
    l = step.get_lower_bound_vec()

    step_data = step.get_matrix_data(k)
    A = step_data['A']
    b = step_data['b']

    y = step.get_output_var()
    u = step.get_input_var()
    u_dim = step.get_input_var_dim()
    n = y.get_dim()
    D = spa.eye(n)

    nonproj_indices = step.nonproj_indices
    A_matrices = []
    b_lvals = []
    b_uvals = []
    psd_cone_handlers = []

    iter_bound_map = handler.iter_bound_map
    yrange = iter_bound_map[y][k]
    urange = map_linstep_to_ranges(y, u, k, handler)

    if len(nonproj_indices) > 0:
        A1D, bl1D, bu1D = equality1D_constraints(D, y, A, u, b, k, handler, indices=step.nonproj_indices)
        A_matrices += A1D
        b_lvals += bl1D
        b_uvals += bu1D

        A2D, bl2D, bu2D, psd_2D = equality2D_constraints(D, y, A, u, b, k, handler, indices=step.nonproj_indices)
        A_matrices += A2D
        b_lvals += bl2D
        b_uvals += bu2D
        # psd_cone_handlers += psd_2D

        C = np.eye(u_dim)
        Across, blcross, bucross, psd_cross = cross_constraints_from_ranges(y.get_dim(), u_dim, handler.problem_dim,
                                                                D.todense(), yrange, A.todense(), urange, b,
                                                                C, urange, C, urange, np.zeros((u_dim, 1)),
                                                                m_indices=step.nonproj_indices,
                                                                n_indices=step.nonproj_indices)
        A_matrices += Across
        b_lvals += blcross
        b_uvals += bucross
        # psd_cone_handlers += psd_cross

    if isinstance(l, Parameter):
        # A_vals, b_lvals, b_uvals = canon_with_l_param(step, k, handler)
        Aproj, b_lproj, b_uproj = canon_with_l_param(step, k, handler)
        lrange = handler.param_bound_map(l)
        psd_cone_handlers.append(
            PSDConeHandler(range_to_list(yrange) + range_to_list(urange) + range_to_list(lrange))
        )
        A_matrices += Aproj
        b_lvals += b_lproj
        b_uvals += b_uproj
    else:
        Aproj, b_lproj, b_uproj = canon_with_l_const(step, k, handler)
        psd_cone_handlers.append(PSDConeHandler(range_to_list(yrange) + range_to_list(urange)))
        A_matrices += Aproj
        b_lvals += b_lproj
        b_uvals += b_uproj

    if handler.add_indiv_RLT:
        ranges = [yrange] + urange
        A_rlt, bl_rlt, bu_rlt = RLT_all_in_range_list(ranges, handler)
        A_matrices += A_rlt
        b_lvals += bl_rlt
        b_uvals += bu_rlt

    return A_matrices, b_lvals, b_uvals, psd_cone_handlers

def canon_with_l_const(step, k, handler):
    y = step.get_output_var()
    u = step.get_input_var()
    l = step.get_lower_bound_vec()
    n = y.get_dim()
    step.get_input_var_dim()
    problem_dim = handler.problem_dim
    iter_bound_map = handler.iter_bound_map
    proj_indices = step.proj_indices
    nonproj_indices = step.nonproj_indices

    A_vals = []
    b_lvals = []
    b_uvals = []

    np.eye(n)
    step_data = step.get_matrix_data(k)
    A = step_data['A']
    b = step_data['b']

    ybounds = iter_bound_map[y][k]
    ubounds = map_linstep_to_ranges(y, u, k, handler)

    yrange1D_handler = RangeHandler1D(ybounds)
    urange1D_handler = RangeHandler1D(ubounds)
    yyTrange_handler = RangeHandler2D(ybounds, ybounds)
    yuTrange_handler = RangeHandler2D(ybounds, ubounds)
    uuTrange_handler = RangeHandler2D(ubounds, ubounds)

    # print(proj_indices)
    # y >= l
    if not handler.add_RLT:
        for i in proj_indices:
            output_mat = spa.lil_matrix((problem_dim, problem_dim))
            insert_vec = np.zeros((n, 1))
            insert_vec[i, 0] = 1
            output_mat[yrange1D_handler.index_matrix()] = insert_vec
            # output_mat[urange1D_handler.index_matrix()] = -insert_vec
            output_mat = (output_mat + output_mat.T) / 2
            A_vals.append(spa.csc_matrix(output_mat))
            b_lvals.append(l[i, 0])
            b_uvals.append(np.inf)

    # y - Au >= b
    for i in proj_indices:
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        insert_vec = np.zeros((n, 1))
        insert_vec[i, 0] = 1
        output_mat[yrange1D_handler.index_matrix()] = insert_vec
        Ai = A[i].reshape((-1, 1))
        output_mat[urange1D_handler.index_matrix()] = -Ai
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(b[i, 0])
        b_uvals.append(np.inf)

    # diag(yyT - ybT - yxTAT - lyT + lxTAT) = diag(-lbT)
    I = np.eye(n)
    for i in proj_indices:
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        Ii = I[i].T.reshape((-1, 1))
        ATj = A.T[:, i].reshape((1, -1))
        bi = b[i, 0]
        li = l[i, 0]
        output_mat[yyTrange_handler.index_matrix()] = Ii @ Ii.T
        output_mat[yrange1D_handler.index_matrix()] = -Ii * bi
        output_mat[yuTrange_handler.index_matrix()] = - Ii @ ATj
        output_mat[yrange1D_handler.index_matrix_horiz()] = - li * Ii.T
        output_mat[urange1D_handler.index_matrix_horiz()] = li * ATj
        output_mat = (output_mat + output_mat.T) / 2
        A_vals.append(spa.csc_matrix(output_mat))
        b_lvals.append(-bi * li)
        b_uvals.append(-bi * li)

    if handler.add_planet:
        y_upper = handler.var_upperbounds[yrange1D_handler.index_matrix()]
        y_lower = handler.var_lowerbounds[yrange1D_handler.index_matrix()]
        handler.var_upperbounds[urange1D_handler.index_matrix()]
        Aub_lower, Aub_upper, _ = get_Aub_bounds(step, k, handler)
        # print(Aub_lower, Aub_upper)
        Au_lower = Aub_lower - b
        Au_upper = Aub_upper - b
        # print(Au_lower, Aub_lower)
        # print(Au_upper, Aub_upper)
        # gaps_vec = (Au_upper - Au_lower).reshape(-1, )
        gaps_vec = np.squeeze(np.asarray(Au_upper - Au_lower))
        pos_gap_indices = np.argwhere(gaps_vec >= 1e-6).reshape(-1, )
        # frac = np.divide((y_upper - y_lower)[pos_gap_indices], (Au_upper - Au_lower)[pos_gap_indices])

        In = np.eye(n)
        for i in range(n):
            if i in nonproj_indices:
                continue
            if i not in pos_gap_indices:
                continue

            mul = (y_upper[i, 0] - y_lower[i, 0]) / (Aub_upper[i, 0] - Aub_lower[i, 0])
            # print(mul)
            Di = mul * A[i]
            ci = mul * (b[i, 0] - Aub_lower[i, 0]) + y_lower[i, 0]

            outmat = spa.lil_matrix((problem_dim, problem_dim))

            # print(Di.shape, Au_upper.shape)
            # print((Di * Au_upper[i, 0]).shape)
            outmat[urange1D_handler.index_matrix()] = (Di * Au_upper[i, 0]).reshape((-1, 1))
            outmat[uuTrange_handler.index_matrix()] = -Di.T @ A[i]
            outmat[urange1D_handler.index_matrix_horiz()] = -ci * A[i]
            outmat[yrange1D_handler.index_matrix()] = -In[i] * Au_upper[i, 0]
            # print(In[i].reshape((-1, 1)) @ A[i])
            outmat[yuTrange_handler.index_matrix()] = In[i].reshape((-1, 1)) @ A[i]
            outmat = (outmat + outmat.T) / 2

            A_vals.append(spa.csc_matrix(outmat))
            b_lvals.append(-ci * Au_upper[i, 0])
            b_uvals.append(np.inf)


    return A_vals, b_lvals, b_uvals


def canon_with_l_param(step, k, handler):
    y = step.get_output_var()
    u = step.get_input_var()
    step.get_lower_bound_vec()
    n = y.get_dim()
    step.get_input_var_dim()
    iter_bound_map = handler.iter_bound_map

    A_vals = []
    b_lvals = []
    b_uvals = []

    np.eye(n)
    step_data = step.get_matrix_data(k)
    step_data['A']
    step_data['b']

    ybounds = iter_bound_map[y][k]
    ubounds = map_linstep_to_ranges(y, u, k, handler)

    RangeHandler1D(ybounds)
    RangeHandler1D(ubounds)
    RangeHandler2D(ybounds, ybounds)
    RangeHandler2D(ybounds, ubounds)
    RangeHandler2D(ubounds, ubounds)

    print('here with l param')
    exit(0)

    return A_vals, b_lvals, b_uvals


def linear_max_proj_bound_canon(step, k, handler):
    u = step.get_input_var()  # remember this is a list of vars
    y = step.get_output_var()

    uranges = map_linstep_to_ranges(y, u, k, handler)

    yrange = handler.iter_bound_map[y][k]
    yrange_handler = RangeHandler1D(yrange)
    RangeHandler1D(uranges)

    l = step.get_lower_bound_vec()
    if not isinstance(l, Parameter):
        l_vec = l.reshape((-1, 1))
        l_lower = l_vec
        l_upper = l_vec
        l_ws = l_vec
    else:
        lrange = handler.param_bound_map[l]
        lrange_handler = RangeHandler1D(lrange)
        l_lower = handler.var_lowerbounds[lrange_handler.index_matrix()]
        l_upper = handler.var_upperbounds[lrange_handler.index_matrix()]
        l_ws = handler.var_warmstart[lrange_handler.index_matrix()]

    # print(l_lower, l_upper)

    proj_indices = np.array(step.proj_indices)
    np.array(step.nonproj_indices)

    Ax_lower, Ax_upper, Ax_ws = get_Aub_bounds(step, k, handler)

    y_lower = Ax_lower.copy()
    y_upper = Ax_upper.copy()
    y_ws = Ax_ws.copy()

    if len(proj_indices) > 0:
        y_lower[proj_indices] = np.maximum(y_lower[proj_indices], l_lower[proj_indices])
        y_upper[proj_indices] = np.maximum(y_upper[proj_indices], l_upper[proj_indices])
        y_ws[proj_indices] = np.maximum(y_ws[proj_indices], l_ws[proj_indices])

    # print('proj')
    # print(y_lower, y_upper, y_ws)
    # print(y, k, y_lower, y_upper)
    # exit(0)

    handler.var_lowerbounds[yrange_handler.index_matrix()] = y_lower
    handler.var_upperbounds[yrange_handler.index_matrix()] = y_upper
    handler.var_warmstart[yrange_handler.index_matrix()] = y_ws


def get_Aub_bounds(step, k, handler):
    u = step.get_input_var()  # remember this is a list of vars
    y = step.get_output_var()

    step_data = step.get_matrix_data(k)
    A = step_data['A'].todense()
    b = step_data['b']

    uranges = map_linstep_to_ranges(y, u, k, handler)
    iter_map = map_linstep_to_iters(y, u, k, handler)

    # print(k, iter_map, uranges)

    # # if thing is iterate 0 or param, use its function, otherwise use lin_bound_map
    A_split = step.split_matrix(A)
    boundaries = step.split_matrix_boundaries()
    # # print(step.split_matrix(DinvA))

    handler.iter_bound_map[y][k]
    urange_handler = RangeHandler1D(uranges)

    u_ws = handler.var_warmstart[urange_handler.index_matrix()]
    l_out = b
    u_out = b
    full_index_mat = urange_handler.index_matrix()

    for i, x in enumerate(u):
        curr_mat = A_split[i]
        curr_bounds = boundaries[i]
        curr_index_mat = (full_index_mat[0][curr_bounds[0]: curr_bounds[1], :], full_index_mat[1])
        l_bound = None
        u_bound = None
        if x.is_param:
            x_set = handler.param_set_map[x]
            if type(x_set) in SET_LINPROP_MAP:
                l_bound, u_bound = SET_LINPROP_MAP[type(x_set)](handler, x, curr_mat)
        else:
            if iter_map[i] == 0: # TODO replace with more general inits than zero
                x_set = handler.iterate_init_set_map[x]
                if type(x_set) in SET_LINPROP_MAP:
                    l_bound, u_bound = SET_LINPROP_MAP[type(x_set)](handler, x, curr_mat)

        if l_bound is None and u_bound is None:
            l_curr = handler.var_lowerbounds[curr_index_mat]
            u_curr = handler.var_upperbounds[curr_index_mat]
            l_bound, u_bound = lin_bound_map(l_curr, u_curr, curr_mat)
        l_out = l_out + l_bound
        u_out = u_out + u_bound

    Ax_lower = l_out
    Ax_upper = u_out
    # print(l_out, u_out)
    Ax_ws = A @ u_ws + b

    return Ax_lower, Ax_upper, Ax_ws


def lin_bound_map(l, u, A):
    # A = A.toarray()
    (m, n) = A.shape
    l_out = np.zeros(m)
    u_out = np.zeros(m)
    for i in range(m):
        lower = 0
        upper = 0
        for j in range(n):
            # if A[i][j] >= 0:
            if A[i, j] >= 0:
                lower += A[i, j] * l[j]
                upper += A[i, j] * u[j]
            else:
                lower += A[i, j] * u[j]
                upper += A[i, j] * l[j]
        l_out[i] = lower
        u_out[i] = upper

    return np.reshape(l_out, (m, 1)), np.reshape(u_out, (m, 1))


def range_to_list(ranges):
    if not isinstance(ranges, list):
        return [ranges]
    else:
        return ranges
