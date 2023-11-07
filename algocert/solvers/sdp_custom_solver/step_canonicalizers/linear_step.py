import numpy as np

from algocert.solvers.sdp_custom_solver.cross_constraints import cross_constraints_from_ranges
from algocert.solvers.sdp_custom_solver.equality_constraints import (
    equality1D_constraints,
    equality2D_constraints,
)
from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D
from algocert.solvers.sdp_custom_solver.RLT_constraints import RLT_ranges
from algocert.solvers.sdp_custom_solver.step_canonicalizers.linear_step_propagation import (
    SET_LINPROP_MAP,
)
from algocert.solvers.sdp_custom_solver.utils import map_linstep_to_iters, map_linstep_to_ranges


def linear_step_canon(step, k, handler):
    step_data = step.get_matrix_data(k)

    # Dstep = step.get_lhs_matrix()
    # Astep = step.get_rhs_matrix()
    # step.get_rhs_const_vec()

    D = step_data['D']
    A = step_data['A']
    b = step_data['b']

    y = step.get_output_var()
    u = step.get_input_var()  # remember this is a stack of variables
    u_dim = step.get_input_var_dim()

    A_matrices, b_lvals, b_uvals = equality1D_constraints(D, y, A, u, b, k, handler)
    psd_cone_handlers = []

    A2D, bl2D, bu2D, psd_2D = equality2D_constraints(D, y, A, u, b, k, handler)

    A_matrices += A2D
    b_lvals += bl2D
    b_uvals += bu2D
    psd_cone_handlers += psd_2D

    iter_bound_map = handler.iter_bound_map
    yrange = iter_bound_map[y][k]
    urange = map_linstep_to_ranges(y, u, k, handler)

    C = np.eye(u_dim)
    Across, blcross, bucross, psd_cross = cross_constraints_from_ranges(y.get_dim(), u_dim, handler.problem_dim,
                                                             D.todense(), yrange, A.todense(), urange, b,
                                                             C, urange, C, urange, np.zeros((u_dim, 1)))
    # print([(h.ranges, h.row_indices) for h in psd_cross])
    # exit(0)
    if handler.add_indiv_RLT:
        for u in urange:
            A_rlt, bl_rlt, bu_rlt = RLT_ranges(yrange, u, handler)
            A_matrices += A_rlt
            b_lvals += bl_rlt
            b_uvals += bu_rlt

    A_matrices += Across
    b_lvals += blcross
    b_uvals += bucross
    psd_cone_handlers += psd_cross

    # print(len(A_matrices), len(b_lvals), len(b_uvals))

    # exit(0)
    return A_matrices, b_lvals, b_uvals, psd_cone_handlers


def linear_step_bound_canon(step, k, handler):
    u = step.get_input_var()  # remember this is a list of vars
    y = step.get_output_var()
    # A = step.get_rhs_matrix()
    # b = step.get_rhs_const_vec()

    step_data = step.get_matrix_data(k)
    A = step_data['A']
    b = step_data['b']

    DinvA = step.solve_linear_system(A.todense())
    Dinvb = step.solve_linear_system(b)

    uranges = map_linstep_to_ranges(y, u, k, handler)
    iter_map = map_linstep_to_iters(y, u, k, handler)
    # print(iter_map)
    # print(handler.iterate_init_set_map)

    # if thing is iterate 0 or param, use its function, otherwise use lin_bound_map
    DinvA_split = step.split_matrix(DinvA)
    boundaries = step.split_matrix_boundaries()
    # print(step.split_matrix(DinvA))

    yrange = handler.iter_bound_map[y][k]
    yrange_handler = RangeHandler1D(yrange)
    urange_handler = RangeHandler1D(uranges)
    # handler.var_lowerbounds[urange_handler.index_matrix()]
    # handler.var_upperbounds[urange_handler.index_matrix()]
    u_ws = handler.var_warmstart[urange_handler.index_matrix()]
    l_out = Dinvb
    u_out = Dinvb
    full_index_mat = urange_handler.index_matrix()

    # for curr_mat, curr_bounds in zip(DinvA_split, boundaries):
    for i in range(len(u)):
        curr_mat = DinvA_split[i]
        curr_bounds = boundaries[i]
        x = u[i]
        curr_index_mat = (full_index_mat[0][curr_bounds[0]: curr_bounds[1], :], full_index_mat[1])
        l_bound = None
        u_bound = None
        if x.is_param:
            x_set = handler.param_set_map[x]
            if type(x_set) in SET_LINPROP_MAP:
                l_bound, u_bound = SET_LINPROP_MAP[type(x_set)](handler, x, curr_mat)
        else:
            if iter_map[i] == 0:  # TODO replace with more general inits than zero
                x_set = handler.iterate_init_set_map[x]
                if type(x_set) in SET_LINPROP_MAP:
                    l_bound, u_bound = SET_LINPROP_MAP[type(x_set)](handler, x, curr_mat)

        if l_bound is None and u_bound is None:
            l_curr = handler.var_lowerbounds[curr_index_mat]
            u_curr = handler.var_upperbounds[curr_index_mat]
            l_bound, u_bound = lin_bound_map(l_curr, u_curr, curr_mat)
        l_out = l_out + l_bound
        u_out = u_out + u_bound

    # print(l_out, u_out)
    y_ws = DinvA @ u_ws + Dinvb
    handler.var_lowerbounds[yrange_handler.index_matrix()] = l_out
    handler.var_upperbounds[yrange_handler.index_matrix()] = u_out
    handler.var_warmstart[yrange_handler.index_matrix()] = y_ws


def lin_bound_map(l, u, A):
    # A = A.toarray()
    (m, n) = A.shape
    l_out = np.zeros(m)
    u_out = np.zeros(m)
    for i in range(m):
        lower = 0
        upper = 0
        for j in range(n):
            if A[i][j] >= 0:
                lower += A[i][j] * l[j]
                upper += A[i][j] * u[j]
            else:
                lower += A[i][j] * u[j]
                upper += A[i][j] * l[j]
        l_out[i] = lower
        u_out[i] = upper

    return np.reshape(l_out, (m, 1)), np.reshape(u_out, (m, 1))
