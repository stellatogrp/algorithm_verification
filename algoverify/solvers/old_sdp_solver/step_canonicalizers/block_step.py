import cvxpy as cp
import numpy as np

from algoverify.solvers.sdp_solver.var_bounds.RLT_constraints import RLT_constraints


def block_step_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars, add_RLT, kwargs):
    step = steps[i]

    # print(step.list_x)
    block_vars = step.list_x
    block_size = step.get_block_size()
    handlers_to_use = []

    u = step.get_output_var()
    u_var = curr.iterate_vars[u].get_cp_var()
    uuT_var = curr.iterate_outerproduct_vars[u]

    u_blocks = []
    for var in block_vars:
        if var.is_param:
            u_blocks.append(param_vars[var].get_cp_var())
        else:
            if curr_or_prev(u, var, iter_id_map) == 0:
                u_blocks.append(prev.iterate_vars[var].get_cp_var())
            else:
                u_blocks.append(curr.iterate_vars[var].get_cp_var())
    # why are these separate for loops ?? just combine them
    for var in block_vars:
        if var.is_param:
            handlers_to_use.append(None)
        else:
            if curr_or_prev(u, var, iter_id_map) == 0:
                handlers_to_use.append(prev)
            else:
                handlers_to_use.append(curr)

    uuT_blocks = [[None for i in range(block_size)] for j in range(block_size)]
    extra_RLT_cons = []

    for i in range(block_size):
        var1 = block_vars[i]
        var1_handler = handlers_to_use[i]
        for j in range(i, block_size):
            var2 = block_vars[j]
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
                if add_RLT:
                    v1_cpvar = var1_handler.iterate_vars[var1].get_cp_var()
                    lower_v1 = var1_handler.iterate_vars[var1].get_lower_bound()
                    upper_v1 = var1_handler.iterate_vars[var1].get_upper_bound()
                    if not var2.is_param:
                        v2_cpvar = var2_handler.iterate_vars[var2].get_cp_var()
                        lower_v2 = var2_handler.iterate_vars[var2].get_lower_bound()
                        upper_v2 = var2_handler.iterate_vars[var2].get_upper_bound()
                    else:
                        v2_cpvar = param_vars[var2].get_cp_var()
                        lower_v2 = param_vars[var2].get_lower_bound()
                        upper_v2 = param_vars[var2].get_upper_bound()
                    extra_RLT_cons += RLT_constraints(cvx_var,
                                                      v1_cpvar, lower_v1, upper_v1, v2_cpvar, lower_v2, upper_v2)

    # print(u)
    # print(uuT_blocks)
    constraints = []
    # exit(0)
    constraints = [
        cp.bmat([
            [uuT_var, u_var],
            [u_var.T, np.array([[1]])]
        ]) >> 0,
        uuT_var == cp.bmat(uuT_blocks),
        u_var == cp.vstack(u_blocks),
    ]
    constraints += extra_RLT_cons

    # bound prop
    lower_u = curr.iterate_vars[u].get_lower_bound()
    upper_u = curr.iterate_vars[u].get_upper_bound()
    if add_RLT:
        constraints += RLT_constraints(uuT_var, u_var, lower_u, upper_u, u_var, lower_u, upper_u)
        constraints += [lower_u <= u_var, u_var <= upper_u]

    return constraints


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

        # if var1_handler != var2_handler:
        #     if var1_id < var2_id:
        #         handler_to_use = var1_handler
        #     else:
        #         handler_to_use = var2_handler
        # else:  # both handlers are the same
        #     handler_to_use = var1_handler
        #
        # out = handler_to_use.iterate_cross_vars[var1][var2]
        # print(var1, var1_id, var2, var2_id, out.shape)
        # if var1_id < var2_id:
        #     return out.T
        # return out


def block_step_bound_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    step = steps[i]
    u = step.get_output_var()

    out_l = []
    out_u = []

    block_vars = step.list_x
    for var in block_vars:
        if var.is_param:
            out_l.append(param_vars[var].get_lower_bound())
            out_u.append(param_vars[var].get_upper_bound())
        else:
            if curr_or_prev(u, var, iter_id_map) == 0:
                # handlers_to_use.append(prev)
                var_bound_obj = prev.iterate_vars[var]
                out_l.append(var_bound_obj.get_lower_bound())
                out_u.append(var_bound_obj.get_upper_bound())
            else:
                # handlers_to_use.append(curr)
                var_bound_obj = curr.iterate_vars[var]
                out_l.append(var_bound_obj.get_lower_bound())
                out_u.append(var_bound_obj.get_upper_bound())
    # for x in out_l:
    #     print(x.shape)
    u_lower_bound = np.vstack(out_l)
    u_upper_bound = np.vstack(out_u)
    # print(var_lower_bound)
    # print(var_upper_bound)
    curr.iterate_vars[u].set_lower_bound(u_lower_bound)
    curr.iterate_vars[u].set_upper_bound(u_upper_bound)


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
