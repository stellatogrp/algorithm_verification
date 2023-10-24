import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler2D


def block_conv_resid_canon(obj, handler):
    problem_dim = handler.problem_dim
    K = handler.K
    iter_bound_map = handler.iter_bound_map
    s = obj.get_iterate()
    A = obj.get_block_mat()
    ATA = (A.T @ A).todense()
    output_mat = np.zeros((problem_dim, problem_dim))

    sK_ranges = map_s_to_ranges(s, iter_bound_map, K)
    sKm1_ranges = map_s_to_ranges(s, iter_bound_map, K-1)
    print(sK_ranges, sKm1_ranges)
    sKouter_rangehandler = RangeHandler2D(sK_ranges, sK_ranges)
    scross_rangehandler = RangeHandler2D(sK_ranges, sKm1_ranges)
    sKm1outer_rangehandler = RangeHandler2D(sKm1_ranges, sKm1_ranges)

    output_mat[sKouter_rangehandler.index_matrix()] = ATA
    output_mat[scross_rangehandler.index_matrix()] = -2 * ATA
    output_mat[sKm1outer_rangehandler.index_matrix()] = ATA

    output_mat = (output_mat + output_mat.T) / 2

    # exit(0)
    return spa.csc_matrix(output_mat)


def map_s_to_ranges(s, iter_bound_map, k):
    ranges = []
    for x in s:
        ranges.append(iter_bound_map[x][k])
    return ranges
