import cvxpy as cp
import numpy as np

from algoverify.solvers.sdp_solver.var_bounds.RLT_constraints import RLT_constraints


def linear_step_canon(steps, i, iteration_handlers, k, iter_id_map, param_vars,
                      param_outerproduct_vars, var_linstep_map, add_RLT, kwargs):
    '''
        Convert a higher level linear step -> a block step followed by a homogenized linear step
    '''
    # print('here')
    constraints = []

    step = steps[i]
    curr = iteration_handlers[k]
    prev = iteration_handlers[k - 1]

    D = step.get_lhs_matrix()
    A = step.get_rhs_matrix()
    A_blocks = step.get_rhs_matrix_blocks()
    b = step.get_rhs_const_vec()
    y = step.get_output_var()
    u = step.get_input_var()  # remember this is a stack of variables

    y_var = curr.iterate_vars[y].get_cp_var()
    yyT_var = curr.iterate_outerproduct_vars[y]
    lower_y = curr.iterate_vars[y].get_lower_bound()
    upper_y = curr.iterate_vars[y].get_upper_bound()
    handlers_to_use = []
    block_size = len(A_blocks)

    # print(step.get_rhs_matrix_blocks())
    # for A in step.get_rhs_matrix_blocks():
    #     print(A.todense())

    u_blocks = []
    lower_u = []
    upper_u = []
    yuT_blocks = []
    for var in u:
        if var.is_param:
            u_blocks.append(param_vars[var].get_cp_var())
            lower_u.append(param_vars[var].get_lower_bound())
            upper_u.append(param_vars[var].get_upper_bound())
            yuT_blocks.append(curr.iterate_param_vars[y][var])
            handlers_to_use.append(None)
        else:
            yuT_blocks.append(curr.iterate_cross_vars[y][var])
            if curr_or_prev(y, var, iter_id_map) == 0:
                u_blocks.append(prev.iterate_vars[var].get_cp_var())
                lower_u.append(prev.iterate_vars[var].get_lower_bound())
                upper_u.append(prev.iterate_vars[var].get_upper_bound())
                handlers_to_use.append(prev)
            else:
                u_blocks.append(curr.iterate_vars[var].get_cp_var())
                lower_u.append(curr.iterate_vars[var].get_lower_bound())
                upper_u.append(curr.iterate_vars[var].get_upper_bound())
                handlers_to_use.append(curr)
    u_var = cp.vstack(u_blocks)
    yuT_var = cp.hstack(yuT_blocks)
    lower_u = np.vstack(lower_u).reshape((-1, 1))
    upper_u = np.vstack(upper_u).reshape((-1, 1))
    extra_RLT_cons = []
    # print(yuT_var.shape)

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
                if add_RLT:
                    # TODO after adding cross, see if this can be removed
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

    # TODO add in cross terms with RHS (if applicable)
    print(f'---k={k}---')
    for x in u:
        if x in var_linstep_map:
            C = var_linstep_map[x].get_lhs_matrix()
            F = var_linstep_map[x].get_rhs_matrix()
            c = var_linstep_map[x].get_rhs_const_vec()
            yxT_var = curr.iterate_cross_vars[y][x]
            # print(linstep_input_var_need_prev(step, iter_id_map))
            # print(linstep_input_var_need_prev(var_linstep_map[x], iter_id_map))
            if curr_or_prev(y, x, iter_id_map) == 0:
                if linstep_input_var_need_prev(step, iter_id_map):
                    if k == 1:
                        print(f'found k = 1, {y}, {x}')
                        continue
                x_handler_indices = get_block_handler_indices(k-1, var_linstep_map[x], iteration_handlers, iter_id_map)
            else:
                x_handler_indices = get_block_handler_indices(k, var_linstep_map[x], iteration_handlers, iter_id_map)
            u_handler_indices = get_block_handler_indices(k, step, iteration_handlers, iter_id_map)
            print(y, x, u_handler_indices, x_handler_indices)

            z_blocks = []
            for i, z in enumerate(var_linstep_map[x].get_input_var()):
                idx = x_handler_indices[i]
                if z.is_param:
                    z_blocks.append(param_vars[z].get_cp_var())
                else:
                    z_blocks.append(iteration_handlers[idx].iterate_vars[z].get_cp_var())
            # print(z_blocks)
            z_var = cp.vstack(z_blocks)
            # print(z_var)

            uxT_var = build_cross_var_mat(u, u_handler_indices, var_linstep_map[x].get_input_var(), x_handler_indices,
                                          iteration_handlers, iter_id_map, param_outerproduct_vars)
            # print((D @ yxT_var @ C.T).shape)
            constraints += [D @ yxT_var @ C.T == A @ uxT_var @ F.T + A @ u_var @ c.T + b @ z_var.T @ F.T + b @ c.T]
            # TODO add PSD constraint to couple y with x
            xxT_var = build_xxT(var_linstep_map[x].get_input_var(), x_handler_indices, iteration_handlers, iter_id_map, param_outerproduct_vars)
            # exit(0)
            # constraints += [
            #     cp.bmat([
            #         [yyT_var, yxT_var, y_var],
            #         [yxT_var.T, xxT_var, var_linstep_map[x].get_output_var()],
            #         [y_var.T, var_linstep_map[x].get_output_var().T, np.array([[1]])]
            #     ]) >> 0,
            # ]
            print(yyT_var.shape, yxT_var.shape, uxT_var.shape, xxT_var.shape, x, var_linstep_map[x].get_output_var().get_dim(), z_var.shape)
            # if curr_or_prev(y, x, iter_id_map) == 0:
            #     x_var = prev.iterate_vars[x].get_cp_var()
            # else:
            #     x_var = curr.iterate_vars[x].get_cp_var()
            # # constraints += [
            # #     cp.bmat([
            # #         [yyT_var, yxT_var, y_var],
            # #         [yxT_var.T, xxT_var, x_var],
            # #         [y_var.T, x_var.T, np.array([[1]])]
            # #     ]) >> 0,
            # # ]
            yzT_var = cp.Variable((yyT_var.shape[0], xxT_var.shape[0]))
            print(yzT_var.shape)
            constraints += [
                cp.bmat([
                    [yyT_var, yzT_var, y_var],
                    [yzT_var.T, xxT_var, z_var],
                    [y_var.T, z_var.T, np.array([[1]])]
                ]) >> 0,
                # D @ yzT_var @ F.T == yxT_var,
            ]
            print((D @ yzT_var @ F.T).shape)
            print(yxT_var.shape)
            print(C.shape)
            print(A.shape)

            # exit(0)


    # exit(0)

    uuT_var = cp.bmat(uuT_blocks)

    constraints += [
        D @ y_var == A @ u_var + b,
        D @ yyT_var @ D.T == A @ uuT_var @ A.T + A @ u_var @ b.T + b @ u_var.T @ A.T + b @ b.T,
        D @ yuT_var == A @ uuT_var + b @ u_var.T,
        cp.bmat([
            [yyT_var, yuT_var, y_var],
            [yuT_var.T, uuT_var, u_var],
            [y_var.T, u_var.T, np.array([[1]])]
        ]) >> 0,
    ]

    if add_RLT:
        constraints += RLT_constraints(uuT_var, u_var, lower_u, upper_u, u_var, lower_u, upper_u)
        constraints += RLT_constraints(yuT_var, y_var, lower_y, upper_y, u_var, lower_u, upper_u)

    constraints += extra_RLT_cons

    return constraints


def get_block_handler_indices(k, step, iteration_handlers, iter_id_map):
    u = step.get_input_var()
    y = step.get_output_var()
    handler_indices = []
    for x in u:
        if x.is_param:
            handler_indices.append(None)
        else:
            if curr_or_prev(y, x, iter_id_map) == 0:
                handler_indices.append(k-1)
            else:
                handler_indices.append(k)
    return handler_indices


def build_cross_var_mat(u, u_handler_indices, x, x_handler_indices, iteration_handlers, iter_id_map, param_outerproduct_vars):
    m = len(u_handler_indices)
    n = len(x_handler_indices)
    print(m, n, u, x)
    out = [[None for j in range(n)] for i in range(m)]
    # print(out)
    for i in range(m):
        for j in range(n):
            var1 = u[i]
            var1_idx = u_handler_indices[i]
            var2 = x[j]
            var2_idx = x_handler_indices[j]
            if var1_idx is None:
                if var2_idx is None:
                    #  TODO fix case of multiple parameters
                    out[i][j] = param_outerproduct_vars[var1]
                else:
                    out[i][j] = iteration_handlers[var2_idx].iterate_param_vars[var2][var1].T
            else:
                if var2_idx is None:
                    out[i][j] = iteration_handlers[var1_idx].iterate_param_vars[var1][var2]
                else:
                    if var1 == var2:
                        if var1_idx == var2_idx:
                            out[i][j] = iteration_handlers[var1_idx].iterate_outerproduct_vars[var1]
                        elif var1_idx > var2_idx:
                            out[i][j] = iteration_handlers[var1_idx].iterate_cross_vars[var1][var1]
                        else:
                            out[i][j] = iteration_handlers[var2_idx].iterate_cross_vars[var1][var1].T
                        # else:
                        #     out[i][j] = get_cross(var1, var2, iteration_handlers[var1_idx], iteration_handlers[var2_idx], iter_id_map)
                    else:
                        out[i][j] = get_cross(var1, var2, iteration_handlers[var1_idx], iteration_handlers[var2_idx], iter_id_map)
                    # print('check', var1, var2, var1_idx, var2_idx, out[i][j])
                    # exit(0)
    print(out)
    # TODO Check that the transposes are correct
    # TODO check what happens if the vars are the same with same idx -> use iterate_outerproduct_vars
    return cp.bmat(out)


def build_xxT(x, x_handler_indices, iteration_handlers, iter_id_map, param_outerproduct_vars):
    print(x)
    print(x_handler_indices)
    block_size = len(x_handler_indices)
    xxT_blocks = [[None for i in range(block_size)] for j in range(block_size)]
    for i in range(block_size):
        var1 = x[i]
        var1_idx = x_handler_indices[i]
        for j in range(i, block_size):
            var2 = x[j]
            var2_idx = x_handler_indices[j]
            if i == j:
                if var1.is_param:
                    xxT_blocks[i][i] = param_outerproduct_vars[var1]
                else:
                    xxT_blocks[i][i] = iteration_handlers[var1_idx].iterate_outerproduct_vars[var1]
            else:
                if var1.is_param:  # TODO cannot handle multiple params
                    if var2.is_param:
                        xxT_blocks[i][j] = param_outerproduct_vars[var1]
                        xxT_blocks[j][i] = param_outerproduct_vars[var1].T
                    else:
                        xxT_blocks[i][j] = iteration_handlers[var2_idx].iterate_param_vars[var2][var1].T
                        xxT_blocks[j][i] = iteration_handlers[var2_idx].iterate_param_vars[var2][var1]
                else:
                    if var2.is_param:
                        xxT_blocks[i][j] = iteration_handlers[var1_idx].iterate_param_vars[var1][var2]
                        xxT_blocks[j][i] = iteration_handlers[var1_idx].iterate_param_vars[var1][var2].T
                    else:
                        cvx_var = get_cross(var1, var2, iteration_handlers[var1_idx], iteration_handlers[var2_idx], iter_id_map)
                        xxT_blocks[i][j] = cvx_var
                        xxT_blocks[j][i] = cvx_var.T
                # cvx_var = get_cross(var1, var2, iteration_handlers[var1_idx], iteration_handlers[var2_idx], iter_id_map)
                # # print(var1, var2, cvx_var.shape)
                # xxT_blocks[i][j] = cvx_var
                # xxT_blocks[j][i] = cvx_var.T
    print(xxT_blocks)
    print(cp.bmat(xxT_blocks))
    return cp.bmat(xxT_blocks)


def get_cross(var1, var2, var1_handler, var2_handler, iter_id_map):
    # print(var1, var2, var1_handler, var2_handler)
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


def linstep_input_var_need_prev(step, iter_id_map):
    '''
        Use to check if a linstep for k relies on any iterates from k - 1
    '''
    y = step.get_output_var()
    u = step.get_input_var()
    # needs_prev = False
    for x in u:
        if not x.is_param:
            if curr_or_prev(y, x, iter_id_map) == 0:
                return True
    else:
        return False


def linear_step_bound_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    step = steps[i]
    #  prev_step = steps[i - 1]
    u = step.get_input_var()  # remember this is a list of vars
    y = step.get_output_var()
    A = step.get_rhs_matrix()
    #  D = step.get_lhs_matrix()
    Dinv = step.get_lhs_matrix_inv()
    b = step.get_rhs_const_vec()

    DinvA = Dinv @ A
    Dinvb = (Dinv @ b).reshape((-1, 1))

    u_lower = []
    u_upper = []
    # print(u)
    for x in u:
        if x.is_param:
            u_lower.append(param_vars[x].get_lower_bound())
            u_upper.append(param_vars[x].get_upper_bound())
        else:
            if curr_or_prev(y, x, iter_id_map) == 0:
                # handlers_to_use.append(prev)
                x_bound_obj = prev.iterate_vars[x]
                u_lower.append(x_bound_obj.get_lower_bound())
                u_upper.append(x_bound_obj.get_upper_bound())
            else:
                # handlers_to_use.append(curr)
                x_bound_obj = curr.iterate_vars[x]
                u_lower.append(x_bound_obj.get_lower_bound())
                u_upper.append(x_bound_obj.get_upper_bound())

    u_lower_bound = np.vstack(u_lower)
    u_upper_bound = np.vstack(u_upper)
    # print(u_lower_bound, u_upper_bound)

    lower_y, upper_y = lin_bound_map(u_lower_bound, u_upper_bound, DinvA)

    # print(lower_y + Dinvb)
    # print(upper_y + Dinvb)

    curr.iterate_vars[y].set_lower_bound(lower_y + Dinvb)
    curr.iterate_vars[y].set_upper_bound(upper_y + Dinvb)


def lin_bound_map(l, u, A):
    A = A.toarray()
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
