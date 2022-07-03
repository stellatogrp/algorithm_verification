import cvxpy as cp
import numpy as np


def block_step_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    step = steps[i]

    # print(step.list_x)
    block_vars = step.list_x
    block_size = step.get_block_size()
    handlers_to_use = []

    u = step.get_output_var()
    u_var = curr.iterate_vars[u]
    uuT_var = curr.iterate_outerproduct_vars[u]

    u_blocks = []
    for var in block_vars:
        if var.is_param:
            u_blocks.append(param_vars[var])
        else:
            if curr_or_prev(u, var, iter_id_map) == 0:
                u_blocks.append(prev.iterate_vars[var])
            else:
                u_blocks.append(curr.iterate_vars[var])

    for var in block_vars:
        if var.is_param:
            handlers_to_use.append(None)
        else:
            if curr_or_prev(u, var, iter_id_map) == 0:
                handlers_to_use.append(prev)
            else:
                handlers_to_use.append(curr)

    uuT_blocks = [[None for i in range(block_size)] for j in range(block_size)]

    for i in range(block_size):
        var1 = block_vars[i]
        var1_handler = handlers_to_use[i]
        for j in range(i, block_size):
            var2 = block_vars[j]
            var2_handler = handlers_to_use[j]

            if i == j:
                if var1.is_param:
                    uuT_blocks[i][i] = param_outerproduct_vars[var1]
                else:
                    uuT_blocks[i][i] = var1_handler.iterate_outerproduct_vars[var1]
            else:
                cvx_var = get_cross(var1, var2, var1_handler, var2_handler, iter_id_map)
                uuT_blocks[i][j] = cvx_var
                uuT_blocks[j][i] = cvx_var.T

    return [
                cp.bmat([
                    [uuT_var, u_var],
                    [u_var.T, np.array([[1]])]
                ]) >> 0,
                uuT_var == cp.bmat(uuT_blocks),
                u_var == cp.vstack(u_blocks),
            ]


def get_cross(var1, var2, var1_handler, var2_handler, iter_id_map):
    if var2.is_param:
        return var1_handler.iterate_param_vars[var1][var2]
    else:  # both are iterates, not parameters
        var1_id = iter_id_map[var1]
        var2_id = iter_id_map[var2]
        if var1_handler != var2_handler:
            if var1_id < var2_id:
                handler_to_use = var1_handler
            else:
                handler_to_use = var2_handler
        else:  # both handlers are the same
            handler_to_use = var1_handler

        out = handler_to_use.iterate_cross_vars[var1][var2]
        if var1_id < var2_id:
            return out.T
        return out


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
