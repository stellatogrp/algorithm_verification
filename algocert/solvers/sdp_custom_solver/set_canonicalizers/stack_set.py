import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.psd_cone_handler import PSDConeHandler
from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D
from algocert.variables.parameter import Parameter


def stack_set_canon(init_set, handler):
    problem_dim = handler.problem_dim
    val_stack = init_set.stack
    x = init_set.get_iterate()
    param_bound_map = handler.param_bound_map
    iter_bound_map = handler.iter_bound_map

    if x.is_param:
        x_range = param_bound_map[x]
    else:
        x_range = iter_bound_map[x][0]

    A_vals = []
    b_lvals = []
    b_uvals = []

    psd_cone_handler_ranges = [x_range]

    xrange1D_handler = RangeHandler1D(x_range)
    xrange1D_handler.index_matrix()
    # print(xrange1D_handler.row_indices)
    curr_xi = 0
    for val in val_stack:
        if isinstance(val, Parameter):
            n = val.get_dim()
            val_range = param_bound_map[val]
            psd_cone_handler_ranges.append(val_range)
            for j in range(val_range[0], val_range[1]):
                x_idx = xrange1D_handler.row_indices[curr_xi]
                outmat_vec = spa.lil_matrix((problem_dim, problem_dim))
                outmat_vec[x_idx, -1] = .5
                outmat_vec[-1, x_idx] = .5
                outmat_vec[j, -1] = -.5
                outmat_vec[-1, j] = -.5
                A_vals.append(spa.csc_matrix(outmat_vec))
                b_lvals.append(0)
                b_uvals.append(0)

                outmat_mat = spa.lil_matrix((problem_dim, problem_dim))
                outmat_mat[x_idx, x_idx] = 1
                outmat_mat[j, j] = -1
                A_vals.append(spa.csc_matrix(outmat_mat))
                b_lvals.append(0)
                b_uvals.append(0)

                curr_xi += 1

        else:
            # print('const')
            n = val[0].shape[0]
            l_val = val[0]
            u_val = val[0]
            for j in range(n):
                l_const = l_val[j]
                u_const = u_val[j]
                x_idx = xrange1D_handler.row_indices[curr_xi]
                # # print(x_idx)
                # # print(val_const)
                # outmat_vec = spa.lil_matrix((problem_dim, problem_dim))
                # outmat_vec[x_idx, -1] = .5
                # outmat_vec[-1, x_idx] = .5
                # A_vals.append(spa.csc_matrix(outmat_vec))
                # b_lvals.append(val_const)
                # b_uvals.append(val_const)

                # outmat_mat = spa.lil_matrix((problem_dim, problem_dim))
                # outmat_mat[x_idx, x_idx] = 1
                # A_vals.append(spa.csc_matrix(outmat_mat))
                # b_lvals.append(val_const ** 2)
                # b_uvals.append(val_const ** 2)

                outmat = spa.lil_matrix((problem_dim, problem_dim))
                outmat[x_idx, x_idx] = 1
                outmat[x_idx, -1] = (-l_const - u_const) / 2
                outmat[-1, x_idx] = (-l_const - u_const) / 2

                A_vals.append(spa.csc_matrix(outmat))
                b_lvals.append(-np.inf)
                b_uvals.append(-l_const * u_const)

                curr_xi += 1

    # print(psd_cone_handler_ranges)
    psd_cones = [PSDConeHandler(psd_cone_handler_ranges)]
    # exit(0)

    return A_vals, b_lvals, b_uvals, psd_cones


def stack_bound_canon(init_set, handler):
    val_stack = init_set.stack
    x = init_set.get_iterate()
    if x.is_param:
        xrange = handler.param_bound_map[x]
    else:
        xrange = handler.iter_bound_map[x][0]

    xrange1D_handler = RangeHandler1D(xrange)
    l_new = []
    u_new = []

    for val in val_stack:
        if isinstance(val, Parameter):
            valrange = handler.param_bound_map[val]
            valrange1D_handler = RangeHandler1D(valrange)
            # l_new.append(handler.param_to_lower_bound_map[x])
            # u_new.append(handler.param_to_upper_bound_map[x])
            # low = handler.var_lowerbounds[]
            # l_new.ap
            l_new.append(handler.var_lowerbounds[valrange1D_handler.index_matrix()])
            u_new.append(handler.var_upperbounds[valrange1D_handler.index_matrix()])
        else:
            l_new.append(val[0].reshape(-1, 1))
            u_new.append(val[1].reshape(-1, 1))

    # print(l_new, u_new)
    l_new = np.vstack(l_new)
    u_new = np.vstack(u_new)
    # print(l_new, u_new)
    handler.var_lowerbounds[xrange1D_handler.index_matrix()] = l_new
    handler.var_upperbounds[xrange1D_handler.index_matrix()] = u_new
