import numpy as np
import scipy.sparse as spa

from algoverify.solvers.sdp_custom_solver.range_handler import RangeHandler1D, RangeHandler2D


def l2_ball_set_canon(init_set, handler):
    '''
        Constraints: (x - c)^T (x - c) <= r ** 2,
        I.e. trace(xx^T - 2xc^T) <= r ** 2 - c^Tc
    '''
    x = init_set.get_iterate()
    r = init_set.r
    c = init_set.c
    n = x.get_dim()
    problem_dim = handler.problem_dim

    if x.is_param:
        param_bound_map = handler.param_bound_map
        xrange = param_bound_map[x]
    else:
        iter_bound_map = handler.iter_bound_map
        xrange = iter_bound_map[x][0]

    # A_vals = []
    # b_lvals = []
    # b_uvals = []

    xrange1D_handler = RangeHandler1D(xrange)
    xrange2D_handler = RangeHandler2D(xrange, xrange)

    outmat = np.zeros((problem_dim, problem_dim))
    outmat[xrange2D_handler.index_matrix()] = np.eye(n)
    outmat[xrange1D_handler.index_matrix()] = -c
    outmat[xrange1D_handler.index_matrix_horiz()] = -c.T

    # note the float() is used to turn the 1x1 c^Tc to a scalar
    return [spa.csc_matrix(outmat)], [-np.inf], [float(r ** 2 - c.T @ c)], []


def l2_ball_bound_canon(init_set, handler):
    r = init_set.r
    c = init_set.c
    u = c + r
    l = c - r

    x = init_set.get_iterate()
    xranges = []

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
        handler.var_lowerbounds[xrange1D_handler.index_matrix()] = l.reshape(-1, 1)
        handler.var_upperbounds[xrange1D_handler.index_matrix()] = u.reshape(-1, 1)
        handler.var_warmstart[xrange1D_handler.index_matrix()] = init_set.sample_point()
