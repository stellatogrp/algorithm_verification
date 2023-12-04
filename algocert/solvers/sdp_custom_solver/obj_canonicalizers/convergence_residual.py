import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler2D
from algocert.solvers.sdp_custom_solver.RLT_constraints import RLT_all_in_range_list


def conv_resid_canon(obj, handler):
    # return output_mat
    problem_dim = handler.problem_dim
    K = handler.K
    iter_bound_map = handler.iter_bound_map

    x = obj.get_iterate()
    x_dim = x.get_dim()
    xK_range = iter_bound_map[x][K]
    xKm1_range = iter_bound_map[x][K-1]
    # print(xK_range, xKm1_range)

    output_mat = np.zeros((problem_dim, problem_dim))

    xK_xKT_range = RangeHandler2D(xK_range, xK_range)
    xKcross_range = RangeHandler2D(xK_range, xKm1_range)
    xKm1_xKm1T_range = RangeHandler2D(xKm1_range, xKm1_range)

    output_mat[xK_xKT_range.index_matrix()] = np.eye(x_dim)
    output_mat[xKm1_xKm1T_range.index_matrix()] = np.eye(x_dim)
    output_mat[xKcross_range.index_matrix()] = -2 * np.eye(x_dim)

    output_mat = (output_mat + output_mat.T) / 2

    # TODO: see if need to add psd constraint

    # test_range = RangeHandler1D([xK_range, xKm1_range])
    # print(test_range.index_matrix())
    # print(output_mat[test_range.index_matrix()])
    # exit(0)

    A_matrices = []
    b_lvals = []
    b_uvals = []

    if handler.add_indiv_RLT:
        ranges = [xK_range, xKm1_range]
        A_rlt, bl_rlt, bu_rlt = RLT_all_in_range_list(ranges, handler)
        A_matrices += A_rlt
        b_lvals += bl_rlt
        b_uvals += bu_rlt

    return spa.csc_matrix(output_mat), A_matrices, b_lvals, b_uvals
