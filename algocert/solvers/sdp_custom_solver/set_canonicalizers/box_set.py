import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D


def box_set_canon(init_set, handler):
    '''
        Constraints: diag(xxT) \leq (l + u) \odot x - l \odot u
        I.e. -inf \leq (M_xx)_i - (l_i + u_i) x_i \leq -l_i * u_i
    '''

    x = init_set.get_iterate()
    l = init_set.l
    u = init_set.u
    problem_dim = handler.problem_dim

    if x.is_param:
        param_bound_map = handler.param_bound_map
        x_range = param_bound_map[x]
    else:
        iter_bound_map = handler.iter_bound_map
        x_range = iter_bound_map[x][0]

    # print(x_range)
    xl, xu = x_range

    A_vals = []
    b_lvals = []
    b_uvals = []

    for j in range(xl, xu):
        curr_mat = np.zeros((problem_dim, problem_dim))
        curr_mat[j, j] = 1
        l_val = l[j - xl, 0]
        u_val = u[j - xl, 0]
        curr_mat[j, -1] = -l_val - u_val
        curr_mat = (curr_mat + curr_mat.T) / 2
        # print(curr_mat)
        A_vals.append(spa.csc_matrix(curr_mat))
        b_lvals.append(-np.inf)
        b_uvals.append(-l_val * u_val)

    # print(b_lvals, b_uvals)

    return A_vals, b_lvals, b_uvals, []


def box_bound_canon(init_set, handler):
    x = init_set.get_iterate()
    #  n = x.get_dim()
    l = init_set.l
    u = init_set.u

    xranges = []

    # if x.is_param:
    #     xrange = handler.param_bound_map[x]
    #     xranges.append(xrange)
    # else:
    #     xrange = handler.iter_bound_map[x][0]

    if x.is_param:
        xrange = handler.param_bound_map[x]
        xranges.append(xrange)
    else:
        # xrange = handler.iter_bound_map[x][0]
        for k in init_set.canon_iter:
            xrange = handler.iter_bound_map[x][k]
            xranges.append(xrange)

    for xrange in xranges:
        xrange1D_handler = RangeHandler1D(xrange)
        # print(handler.var_lowerbounds[xrange1D_handler.index_matrix()])
        handler.var_lowerbounds[xrange1D_handler.index_matrix()] = l
        handler.var_upperbounds[xrange1D_handler.index_matrix()] = u
        handler.var_warmstart[xrange1D_handler.index_matrix()] = init_set.sample_point()
