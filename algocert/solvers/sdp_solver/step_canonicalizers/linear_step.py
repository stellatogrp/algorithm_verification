import cvxpy as cp
import numpy as np

from algocert.solvers.sdp_solver.var_bounds.RLT_constraints import RLT_constraints


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
    handlers_to_use = []
    block_size = len(A_blocks)

    print(step.get_rhs_matrix_blocks())
    # for A in step.get_rhs_matrix_blocks():
    #     print(A.todense())

    u_blocks = []
    yuT_blocks = []
    for var in u:
        if var.is_param:
            u_blocks.append(param_vars[var].get_cp_var())
            yuT_blocks.append(curr.iterate_param_vars[y][var])
            handlers_to_use.append(None)
        else:
            yuT_blocks.append(curr.iterate_cross_vars[y][var])
            if curr_or_prev(y, var, iter_id_map) == 0:
                u_blocks.append(prev.iterate_vars[var].get_cp_var())
                handlers_to_use.append(prev)
            else:
                u_blocks.append(curr.iterate_vars[var].get_cp_var())
                handlers_to_use.append(curr)
    u_var = cp.vstack(u_blocks)
    yuT_var = cp.hstack(yuT_blocks)
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
    # TODO add RLT for blocks with themselves

    uuT_var = cp.bmat(uuT_blocks)

    constraints += [
        D @ y_var == A @ u_var + b,
        D @ yyT_var @ D.T == A @ uuT_var @ A.T + A @ u_var @ b.T + b @ u_var.T @ A.T + b @ b.T,
        cp.bmat([
            [yyT_var, yuT_var, y_var],
            [yuT_var.T, uuT_var, u_var],
            [y_var.T, u_var.T, np.array([[1]])]
        ]) >> 0,
    ]

    constraints += extra_RLT_cons

    # exit(0)
    # D = step.get_lhs_matrix()
    # Dinv = step.get_lhs_matrix_inv()
    # A = step.get_rhs_matrix()
    # b = step.get_rhs_const_vec()
    # y = step.get_output_var()
    # u = step.get_input_var()
    # u_dim = 0
    # for x in u:
    #     u_dim += x.get_dim()
    # name = y.get_name()
    # new_name = name + '_block'
    # u_block = Iterate(u_dim, name=new_name)
    # step1 = BlockStep(u_block, u)
    # step2 = BasicLinearStep(y, u_block, A=A, D=D, b=b, Dinv=Dinv)
    # return [u_block], [step1, step2]
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
