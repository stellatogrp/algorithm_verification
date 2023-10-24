import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.psd_cone_handler import PSDConeHandler
from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D, RangeHandler2D
from algocert.solvers.sdp_custom_solver.utils import map_linstep_to_ranges


def cross_constraint_steps(step1, k1, step2, k2, handler):
    # D = step.get_lhs_matrix()
    # A = step.get_rhs_matrix()
    # step.get_rhs_matrix_blocks()
    # b = step.get_rhs_const_vec()
    # y = step.get_output_var()
    # u = step.get_input_var()
    step1.get_lhs_matrix()
    step1.get_rhs_matrix()
    step1.get_rhs_const_vec()
    step1.get_output_var()
    step1.get_input_var()

    step2.get_lhs_matrix()
    step2.get_rhs_matrix()
    step2.get_rhs_const_vec()
    step2.get_output_var()
    step2.get_input_var()

    return [], [], [], []


def cross_constraints(D, y, A, u, b, C, z, F, x, c, k1, k2):
    print(D)
    return [], [], [], []


def range_to_list(ranges):
    if not isinstance(ranges, list):
        return [ranges]
    else:
        return ranges


def cross_constraints_from_ranges(m, n, problem_dim,
                                  D, y_range, A, u_range, b,
                                  C, z_range, F, x_range, c,
                                  only_include_psd_cones=False):
    '''
        Assumes D, A, C, F are dense
    '''
    # print(D)
    # print(y_range)
    # print(A)
    # print(u_range)
    # print(b)

    # print(C)
    # print(z_range)
    # print(F)
    # print(x_range)
    # print(x)

    if spa.issparse(D):
        D = D.todense()
    if spa.issparse(A):
        A = A.todense()
    if spa.issparse(C):
        C = C.todense()
    if spa.issparse(F):
        F = F.todense()

    A_matrices = []
    b_lvals = []
    b_uvals = []
    psd_cone_handlers = []  # use to create PSD cone objects

    bcT = b @ c.T
    # print(bcT.shape, m, n)

    yzrange_handler = RangeHandler2D(y_range, z_range)
    uxrange_handler = RangeHandler2D(u_range, x_range)
    urange_handler = RangeHandler1D(u_range)
    xrange_handler = RangeHandler1D(x_range)
    RangeHandler1D(y_range)
    RangeHandler1D(z_range)

    # if not isinstance(y_range, list):
    #     yrange_list = [y_range]
    # else:
    #     yrange_list = y_range
    # if not isinstance(z_range, list):
    #     zrange_list = [z_range]
    # else:
    #     zrange_list = z_range
    yrange_list = range_to_list(y_range)
    zrange_list = range_to_list(z_range)
    urange_list = range_to_list(u_range)
    range_to_list(x_range)

    h = PSDConeHandler(yrange_list + urange_list + zrange_list)
    psd_cone_handlers.append(h)
    # print(yrange_list, zrange_list, urange_list, xrange_list)
    # print(h.row_indices)
    # print(list(set(h.row_indices)))
    # exit(0)

    if only_include_psd_cones:
        return A_matrices, b_lvals, b_uvals, psd_cone_handlers

    for i in range(m):
        for j in range(n):
            # print(i, j)
            # outmat = np.zeros((problem_dim, problem_dim))
            outmat = spa.lil_matrix((problem_dim, problem_dim))
            Di = D[i].T.reshape((-1, 1))
            CTj = C.T[:, j].T.reshape((1, -1))
            # print(Di, CTj)
            Ai = A[i].T.reshape((-1, 1))
            FTj = F.T[:, j].T.reshape((1, -1))
            bi = b[i, 0]
            cj = c[j, 0]

            # print(Di.shape, CTj.shape)
            outmat[yzrange_handler.index_matrix()] = Di @ CTj
            outmat[uxrange_handler.index_matrix()] = -Ai @ FTj
            outmat[urange_handler.index_matrix()] = -Ai * cj
            outmat[xrange_handler.index_matrix_horiz()] = -bi * FTj

            outmat = (outmat + outmat.T) / 2
            outmat = spa.csc_matrix(outmat)
            A_matrices.append(outmat)
            b_lvals.append(bcT[i, j])
            b_uvals.append(bcT[i, j])

    return A_matrices, b_lvals, b_uvals, psd_cone_handlers


def cross_constraints_between_linsteps(y1, y2, k1, k2, handler,
                                       only_include_psd_cones=False):
    if k1 == 0 or k2 == 0:
        return [], [], [], []

    var_linstep_map = handler.var_linstep_map
    step1 = var_linstep_map[y1]
    step2 = var_linstep_map[y2]

    step1_data = step1.get_matrix_data(k1)
    # D1 = step1.get_lhs_matrix()
    # A1 = step1.get_rhs_matrix()
    # b1 = step1.get_rhs_const_vec()
    D1 = step1_data['D']
    A1 = step1_data['A']
    b1 = step1_data['b']
    u1 = step1.get_input_var()
    u1ranges = map_linstep_to_ranges(y1, u1, k1, handler)
    y1range = handler.iter_bound_map[y1][k1]
    # print(u1ranges)

    step2_data = step2.get_matrix_data(k2)
    # D2 = step2.get_lhs_matrix()
    # A2 = step2.get_rhs_matrix()
    # b2 = step2.get_rhs_const_vec()
    D2 = step2_data['D']
    A2 = step2_data['A']
    b2 = step2_data['b']
    u2 = step2.get_input_var()
    u2ranges = map_linstep_to_ranges(y2, u2, k2, handler)
    y2range = handler.iter_bound_map[y2][k2]
    # print(u2ranges)

    m = y1.get_dim()
    n = y2.get_dim()

    return cross_constraints_from_ranges(m, n, handler.problem_dim,
                                         D1, y1range, A1, u1ranges, b1,
                                         D2, y2range, A2, u2ranges, b2,
                                         only_include_psd_cones=only_include_psd_cones)


def cross_constraints_linstep_to_not(y1, y2, k1, k2, handler):
    if k1 == 0 or k2 == 0:
        return [], [], [], []
    var_linstep_map = handler.var_linstep_map
    step1 = var_linstep_map[y1]

    # D1 = step1.get_lhs_matrix()
    # A1 = step1.get_rhs_matrix()
    # b1 = step1.get_rhs_const_vec()
    step1_data = step1.get_matrix_data(k1)
    D1 = step1_data['D']
    A1 = step1_data['A']
    b1 = step1_data['b']
    u1 = step1.get_input_var()
    u1ranges = map_linstep_to_ranges(y1, u1, k1, handler)
    y1range = handler.iter_bound_map[y1][k1]

    y2_dim = y2.get_dim()
    if y2.is_param:
        y2range = handler.param_bound_map[y2]
    else:
        y2range = handler.iter_bound_map[y2][k2]
    C = np.eye(y2_dim)

    # Across, blcross, bucross = cross_constraints_from_ranges(y.get_dim(), u_dim, handler.problem_dim,
    #                                                          D.todense(), yrange, A.todense(), urange, b,
    #                                                          C, urange, C, urange, np.zeros((u_dim, 1)))
    # exit(0)
    return cross_constraints_from_ranges(y1.get_dim(), y2_dim, handler.problem_dim,
                                         D1, y1range, A1, u1ranges, b1,
                                         C, y2range, C, y2range, np.zeros((y2_dim, 1)))
