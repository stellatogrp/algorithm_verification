import cvxpy as cp
import numpy as np

from algocert.solvers.sdp_solver.var_bounds.RLT_constraints import RLT_constraints


def nonneg_orthant_proj_canon(steps, i, iteration_handlers, k, iter_id_map, param_vars, param_outerproduct_vars,
                              var_linstep_map, add_RLT, kwargs):
    step = steps[i]
    steps[i-1]
    curr = iteration_handlers[k]
    # prev = iteration_handlers[k - 1]

    y = step.get_output_var()
    x = step.get_input_var()
    n = x.get_dim()

    y_var = curr.iterate_vars[y].get_cp_var()
    yyT_var = curr.iterate_outerproduct_vars[y]
    x_var = curr.iterate_vars[x].get_cp_var()
    xxT_var = curr.iterate_outerproduct_vars[x]

    yxT_var = curr.iterate_cross_vars[y][x]

    constraints = [y_var >= 0, yyT_var >= 0, y_var >= x_var, cp.diag(yyT_var - yxT_var) == 0]
    constraints += [
        cp.bmat([
            [yyT_var, yxT_var, y_var],
            [yxT_var.T, xxT_var, x_var],
            [y_var.T, x_var.T, np.array([[1]])]
        ]) >> 0,
    ]

    # print(var_linstep_map)
    if x in var_linstep_map:
        # print(x)
        # print('in linstep map')
        step = var_linstep_map[x]
        D = step.get_lhs_matrix()
        A = step.get_rhs_matrix()
        # A_blocks = step.get_rhs_matrix_blocks()
        b = step.get_rhs_const_vec()
        step.get_input_var()

        u_var, yuT_var, uuT_var, extra_RLT_cons = return_blocked_vars(k, y, x, iteration_handlers, step, param_vars, param_outerproduct_vars, iter_id_map, add_RLT)

        constraints += [
            yxT_var @ D.T == yuT_var @ A.T + y_var @ b.T,
            cp.bmat([
                [yyT_var, yuT_var, y_var],
                [yuT_var.T, uuT_var, u_var],
                [y_var.T, u_var.T, np.array([[1]])]
            ]) >> 0,
        ]

        constraints += extra_RLT_cons

    lower_y = curr.iterate_vars[y].get_lower_bound()
    upper_y = curr.iterate_vars[y].get_upper_bound()
    lower_x = curr.iterate_vars[x].get_lower_bound()
    upper_x = curr.iterate_vars[x].get_upper_bound()

    if add_RLT:
        extra_constraints = RLT_constraints(yxT_var, y_var, lower_y, upper_y, x_var, lower_x, upper_x)
        constraints += extra_constraints

        # constraints += RLT_constraints(xxT_var, x_var, lower_x, upper_x, x_var, lower_x, upper_x)
        # constraints += [lower_x <= x_var, x_var <= upper_x]
        # constraints += RLT_constraints(yyT_var, y_var, lower_y, upper_y, y_var, lower_y, upper_y)
        # constraints += [lower_y <= y_var, y_var <= upper_y]

        if 'add_planet' in kwargs:
            if kwargs['add_planet']:
                test = True
                if test:
                    print((upper_x - lower_x).reshape(-1,))
                    gaps_vec = (upper_x - lower_x).reshape(-1,)
                    pos_gap_indices = np.argwhere(gaps_vec >= 1e-5).reshape(-1, )
                    zero_gap_indices = np.argwhere(gaps_vec < 1e-5).reshape(-1, )
                    print(pos_gap_indices, zero_gap_indices)

                    frac = np.divide((upper_y - lower_y)[pos_gap_indices], (upper_x - lower_x)[pos_gap_indices])
                    n_pos = pos_gap_indices.shape[0]
                    A = np.zeros((n_pos, n_pos))
                    # print(A)
                    for i in range(n_pos):
                        A[i, i] = frac[i, 0]
                    b = np.multiply(frac, -lower_x[pos_gap_indices]) + lower_y[pos_gap_indices]

                    # print(xxT_var.shape)
                    # print(xxT_var[pos_gap_indices].shape)
                    # print(xxT_var[pos_gap_indices][:, pos_gap_indices].shape)
                    # print(xxT_var[pos_gap_indices][:, pos_gap_indices])
                    # print(xxT_var[pos_gap_indices, pos_gap_indices].shape)
                    # exit(0)

                    constr = A @ x_var[pos_gap_indices] @ upper_x[pos_gap_indices].T - \
                        A @ xxT_var[pos_gap_indices][:, pos_gap_indices] + \
                        b @ upper_x[pos_gap_indices].T - \
                        b @ x_var[pos_gap_indices].T - \
                        y_var[pos_gap_indices] @ upper_x[pos_gap_indices].T + \
                        yxT_var[pos_gap_indices][:, pos_gap_indices]

                    # print(constr, constr.shape)
                    constraints += [constr >= 0]
                    # print(lower_x[zero_gap_indices])
                    # exit(0)

                    # A = np.diag(frac)
                    # print(frac)
                    # print(A, A.shape)

                    # for idx in zero_gap_indices:
                    #     # constraints += [x_var[idx] == upper_x[idx]]
                    #     constraints += [y_var[idx] == np.maximum(upper_x[idx], 0)]
                    #     constraints += [yyT_var[idx, idx] == (np.maximum(upper_x[idx], 0)) ** 2]

                    # TODO: only planet for pos_gap_indices, for the rest, we should just be able to leave them alone?
                    # exit(0)
                else:
                    # frac = np.divide((upper_y - lower_y)[pos_gap_indices], (upper_x - lower_x)[pos_gap_indices])
                    frac = np.divide(upper_y - lower_y, upper_x - lower_x)
                    # n = pos_gap_indices.shape[0]
                    A = np.zeros((n, n))
                    for i in range(n):
                        # this absolutely should not have been necessary, but I couldn't get the numpy functions
                        # to work as desired
                        A[i, i] = frac[i, 0]
                    b = np.multiply(frac, -lower_x) + lower_y

                    # exit(0)

                    # constraints += [
                    #     y_var <= A @ x_var + b,
                    #     yyT_var <= A @ xxT_var @ A.T + A @ x_var @ b.T + b @ x_var.T @ A.T + b @ b.T
                    # ]

                    constraints += [
                        cp.diag(A @ x_var @ upper_x.T - A @ xxT_var + b @ upper_x.T -
                                b @ x_var.T - y_var @ upper_x.T + yxT_var) >= 0,

                        # cp.diag()

                        # cp.diag(A @ xxT_var - A @ x_var @ lower_x.T + b @ x_var.T - b @ lower_x.T \
                        # - yxT_var + y_var @ lower_x.T) >= 0,
                        # cp.diag(
                        #     A @ xxT_var + A @ x_var @ b.T - A @ yxT_var.T + b @ x_var.T @ A.T + b @ b.T \
                        #  - b @ y_var.T - yxT_var @ A.T - y_var @ b.T + yyT_var
                        # ) >= 0,
                    ]
    return constraints


def return_blocked_vars(k, y, x, iteration_handlers, step,
                        param_vars, param_outerproduct_vars, iter_id_map,
                        add_RLT):
    '''
        Returns u_var, uuT_var, and yuT_var where u is the blocks for x
        Assumes the handler for y and x are both curr
    '''
    curr = iteration_handlers[k]
    prev = iteration_handlers[k - 1]
    u = step.get_input_var()
    handlers_to_use = []
    block_size = len(step.get_rhs_matrix_blocks())
    u_blocks = []
    yuT_blocks = []
    extra_RLT_cons = []
    y_var = curr.iterate_vars[y].get_cp_var()
    y_l = curr.iterate_vars[y].get_lower_bound()
    y_u = curr.iterate_vars[y].get_upper_bound()

    for var in u:
        if var.is_param:
            u_blocks.append(param_vars[var].get_cp_var())
            yuT_blocks.append(curr.iterate_param_vars[y][var])
            handlers_to_use.append(None)
        else:
            yvar_T = curr.iterate_cross_vars[y][var]
            yuT_blocks.append(yvar_T)
            if curr_or_prev(y, var, iter_id_map) == 0:
                prev_var = prev.iterate_vars[var].get_cp_var()
                prev_l = prev.iterate_vars[var].get_lower_bound()
                prev_u = prev.iterate_vars[var].get_upper_bound()

                if add_RLT:
                    extra_RLT_cons += RLT_constraints(yvar_T, y_var, y_l, y_u, prev_var, prev_l, prev_u)

                # y_var, lower_y, upper_y, x_var, lower_x, upper_x)

                u_blocks.append(prev_var)
                handlers_to_use.append(prev)
            else:
                curr_var = curr.iterate_vars[var].get_cp_var()
                curr_l = curr.iterate_vars[var].get_lower_bound()
                curr_u = curr.iterate_vars[var].get_upper_bound()

                if add_RLT:
                    extra_RLT_cons += RLT_constraints(yvar_T, y_var, y_l, y_u, curr_var, curr_l, curr_u)

                u_blocks.append(curr_var)
                handlers_to_use.append(curr)
    u_var = cp.vstack(u_blocks)
    yuT_var = cp.hstack(yuT_blocks)

    uuT_blocks = [[None for i in range(block_size)] for j in range(block_size)]
    for i in range(block_size):
        var1 = u[i]
        var1_handler = handlers_to_use[i]
        for j in range(i, block_size):
            var2 = u[j]
            var2_handler = handlers_to_use[j]

            if i == j:
                if var1.is_param:
                    uuT_blocks[i][i] = param_outerproduct_vars[var1]
                else:
                    uuT_blocks[i][i] = var1_handler.iterate_outerproduct_vars[var1]
            else:
                cvx_var = get_cross(var1, var2, var1_handler, var2_handler, iter_id_map)
                # print(var1, var2, cvx_var.shape)
                uuT_blocks[i][j] = cvx_var
                uuT_blocks[j][i] = cvx_var.T
    uuT_var = cp.bmat(uuT_blocks)

    return u_var, yuT_var, uuT_var, extra_RLT_cons


def get_cross(var1, var2, var1_handler, var2_handler, iter_id_map):
    if var2.is_param:
        return var1_handler.iterate_param_vars[var1][var2]
    else:  # both are iterates, not parameters
        var1_id = iter_id_map[var1]
        var2_id = iter_id_map[var2]
        if var1_handler == var2_handler:
            if var1_id < var2_id:
                return var1_handler.iterate_cross_vars[var2][var1].T
            else:
                return var1_handler.iterate_cross_vars[var1][var2]
        else:  # diff handlers, use the earlier var handler
            if var1_id < var2_id:
                return var1_handler.iterate_cross_vars[var1][var2]
            else:
                return var2_handler.iterate_cross_vars[var2][var1].T


def curr_or_prev(var1, var2, iter_id_map):
    """
    Returning 0 means to use the previous iteration handler for var2
    Returning 1 means to use the current
    """
    i1 = iter_id_map[var1]
    i2 = iter_id_map[var2]
    if i1 <= i2:
        return 0
    return 1


def nonneg_orthant_proj_bound_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    step = steps[i]
    #  prev_step = steps[i - 1]
    y = step.get_output_var()
    x = step.get_input_var()
    n = y.get_dim()
    lower_x = curr.iterate_vars[x].get_lower_bound()
    upper_x = curr.iterate_vars[x].get_upper_bound()
    # print(lower_x, upper_x)
    zeros = np.zeros((n, 1))
    lower_y = np.maximum(lower_x, zeros)
    upper_y = np.maximum(upper_x, zeros)
    # print(lower_y, upper_y)
    curr.iterate_vars[y].set_lower_bound(lower_y)
    curr.iterate_vars[y].set_upper_bound(upper_y)
