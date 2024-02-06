import numpy as np

from algoverify.init_set.l2_ball_set import L2BallSet
from algoverify.init_set.stack_set import StackSet
from algoverify.solvers.sdp_custom_solver.holder_bounds import holder_bounds
from algoverify.solvers.sdp_custom_solver.range_handler import RangeHandler1D
from algoverify.variables.parameter import Parameter

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


def stack_set_linprop(handler, x, A, b=None):
    if x.is_param:
        x_set = handler.param_set_map[x]
    else:
        x_set = handler.iterate_init_set_map[x]

    m = A.shape[0]
    if b is None:
        b = np.zeros((m, 1))

    l_out = b
    u_out = b

    val_stack = x_set.stack
    curr_A_left = 0
    for val in val_stack:
        if isinstance(val, Parameter):
            n = val.get_dim()
            curr_A_right = curr_A_left + n
            A_curr = A[:, curr_A_left: curr_A_right]

            val_set = handler.param_set_map[val]
            if type(val_set) in SET_LINPROP_MAP:
                l_curr, u_curr = SET_LINPROP_MAP[type(val_set)](handler, val, A_curr)
            else:
                valrange = handler.param_bound_map[val]
                valrange1D_handler = RangeHandler1D(valrange)
                l_val = handler.var_lowerbounds[valrange1D_handler.index_matrix()]
                u_val = handler.var_upperbounds[valrange1D_handler.index_matrix()]
                l_curr, u_curr = lin_bound_map(l_val, u_val, A_curr)
        else:
            l_val = val[0]
            u_val = val[1]
            n = l_val.shape[0]
            curr_A_right = curr_A_left + n
            A_curr = A[:, curr_A_left: curr_A_right]
            l_curr, u_curr = lin_bound_map(l_val, u_val, A_curr)

        l_out = l_out + l_curr
        u_out = u_out + u_curr
        curr_A_left = curr_A_right

    return l_out, u_out


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
    StackSet: stack_set_linprop,
}
