import numpy as np

from algocert.init_set.l2_ball_set import L2BallSet
from algocert.solvers.sdp_custom_solver.holder_bounds import holder_bounds

'''
    A collection of functions to propagate linear bounds Ax + b
    where x lives in some specific parameter set
'''


def l2_ball_set_linprop(handler, x, A, b=None):
    if x.is_param:
        param_set = handler.param_set_map[x]
    else:
        param_set = handler.iterate_init_set_map[x]  # TODO rename param_set to something more general
    c = param_set.c
    r = param_set.r
    m = A.shape[0]
    if b is None:
        b = np.zeros((m, 1))
    # l, u = holder_bounds(p, x0, r, Al, bl)
    # l, u = holder_bounds(2, c, r, A, b)
    return holder_bounds(2, c, r, A, b)


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


SET_LINPROP_MAP = {
    L2BallSet: l2_ball_set_linprop,
}
