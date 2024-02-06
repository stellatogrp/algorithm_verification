import cvxpy as cp
import numpy as np


def nonneg_lin_canon(steps, i, iteration_handlers, k, iter_id_map, param_vars, param_outerproduct_vars, add_RLT, kwargs):
    step = steps[i]
    curr = iteration_handlers[k]
    prev = iteration_handlers[k - 1]

    # block_vars = step.list_x
    y = step.get_output_var()
    u = step.get_input_var()  # remember this is a stack of variables
    b = step.get_rhs_const_vec()
    C = step.get_rhs_matrix()
    C_blocks = step.C_blocks
    block_size = len(C_blocks)
    assert block_size == len(u)
    handlers_to_use = []

    y_var = curr.iterate_vars[y].get_cp_var()
    yyT_var = curr.iterate_outerproduct_vars[y]
    u_blocks = []
    yuT_blocks = []
    uuT_blocks = [[None for i in range(block_size)] for j in range(block_size)]

    # first build u, yuT, and the handlers that are needed for the cross terms
    for i in range(block_size):
        var = u[i]
        if var.is_param:
            handlers_to_use.append(None)
            u_blocks.append(param_vars[var].get_cp_var())
            yuT_blocks.append(curr.iterate_param_vars[y][var])
        else:
            yuT_blocks.append(curr.iterate_cross_vars[y][var])
            if curr_or_prev(y, var, iter_id_map) == 0:
                handlers_to_use.append(prev)
                u_blocks.append(prev.iterate_vars[var].get_cp_var())
            else:
                handlers_to_use.append(curr)
                u_blocks.append(curr.iterate_vars[var].get_cp_var())
    # print(yuT_blocks)

    for i in range(block_size):
        var1 = u[i]
        var1_handler = handlers_to_use[i]
        for j in range(i, block_size):
            var2 = u[j]
            var2_handler = handlers_to_use[j]
            # print(var1, var2)
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
    # print(uuT_blocks)

    u_var = cp.vstack(u_blocks)
    yuT_var = cp.hstack(yuT_blocks)
    uuT_var = cp.bmat(uuT_blocks)
    constraints = [
        y_var >= 0,
        yyT_var >= 0,
        y_var >= C @ u_var + b,
        cp.diag(yyT_var - yuT_var @ C.T - y_var @ b.T) == 0,
        cp.bmat([
            [yyT_var, yuT_var, y_var],
            [yuT_var.T, uuT_var, u_var],
            [y_var.T, u_var.T, np.array([[1]])]
        ]) >> 0,
    ]

    # TODO: add RLT

    return constraints


def nonneg_lin_bound_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    pass


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
