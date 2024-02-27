import numpy as np


def linear_step_canon(step, model, k, iter_to_gp_var_map, param_to_gp_var_map, iter_to_id_map):
    """
    canonicalizes the iteration
    D y = A u + b

    u can be a block, so you can think of it as
    D y = [A_1, ..., A_j][u_1] + b
                         ....
                         [u_j]

    We partition the A matrix into its block form
        based on the dimensions of u_1, ..., u_j

    iter_to_gp_var_map and param_to_gp_var_map are needed to get the gurobi vars

    iter_to_id_map is needed to check if we want the current or previous u_i's
    """
    step_data = step.get_matrix_data(k)
    # D = step.get_lhs_matrix()
    # A = step.get_rhs_matrix()
    # b = step.get_rhs_const_vec()
    D = step_data['D']
    A = step_data['A']
    b = step_data['b']

    y = step.get_output_var()
    u = step.get_input_var()  # remember that this is a block of variables

    y_var = iter_to_gp_var_map[y]
    constraint_lhs = D @ y_var[k]
    constraint_rhs = 0

    # create the boundaries to partition the A matrix
    left = 0
    right = 0
    boundaries = []
    for x in u:
        n = x.get_dim()
        right = left + n
        # print(left, right)
        # print(A.tocsc()[:, left: right].shape)
        boundaries.append((left, right))
        left = right
    # print(boundaries)

    u_idx = map_linstep_to_iters(y, u, k, iter_to_id_map)
    for i, x in enumerate(u):
        (left, right) = boundaries[i]
        idx = u_idx[i]
        if x.is_param:
            # print(x)
            x_var = param_to_gp_var_map[x]
        else:
            x_varmatrix = iter_to_gp_var_map[x]
            # if iter_to_id_map[y] <= iter_to_id_map[x]:
            #     x_var = x_varmatrix[k-1]
            # else:
            #     x_var = x_varmatrix[k]
            x_var = x_varmatrix[idx]

        constraint_rhs += A.tocsc()[:, left: right] @ x_var

    # print('rhs', constraint_rhs.shape)
    # print('lhs', constraint_lhs.shape)
    # print(b, b.shape)
    # constraint_rhs += b.reshape(-1, )
    b = b.reshape(-1, )
    # exit(0)
    constraint_rhs += b.reshape(constraint_rhs.shape)

    model.addConstr(constraint_lhs == constraint_rhs)


def linear_step_bound_canon(step, k, iter_to_id_map,
                            iter_to_lower_bound_map, iter_to_upper_bound_map,
                            param_to_lower_bound_map, param_to_upper_bound_map):
    u = step.get_input_var()  # remember this is a list of vars
    y = step.get_output_var()
    # Dinv = step.get_lhs_matrix_inv()

    step_data = step.get_matrix_data(k)
    # A = step.get_rhs_matrix()
    #  D = step.get_lhs_matrix()
    # b = step.get_rhs_const_vec()
    A = step_data['A']
    b = step_data['b']

    # DinvA = Dinv @ A
    # Dinvb = (Dinv @ b).reshape((-1, 1))

    DinvA = step.solve_linear_system(A.todense())
    Dinvb = step.solve_linear_system(b)
    u_lower = []
    u_upper = []

    u_idx = map_linstep_to_iters(y, u, k, iter_to_id_map)

    # for x in u:
    #     if x.is_param:
    #         u_lower.append(param_to_lower_bound_map[x])
    #         u_upper.append(param_to_upper_bound_map[x])
    #         # print(param_to_lower_bound_map[x].shape)
    #     else:
    #         x_lowermat = iter_to_lower_bound_map[x]
    #         x_uppermat = iter_to_upper_bound_map[x]
    #         if iter_to_id_map[y] <= iter_to_id_map[x]:
    #             x_lower = x_lowermat[k - 1]
    #             x_upper = x_uppermat[k - 1]
    #         else:
    #             x_lower = x_lowermat[k]
    #             x_upper = x_uppermat[k]
    #         u_lower.append(x_lower)
    #         u_upper.append(x_upper)

    for idx, x in zip(u_idx, u):
        if idx is None:
            u_lower.append(param_to_lower_bound_map[x])
            u_upper.append(param_to_upper_bound_map[x])
        else:
            x_lowermat = iter_to_lower_bound_map[x]
            x_uppermat = iter_to_upper_bound_map[x]
            u_lower.append(x_lowermat[idx])
            u_upper.append(x_uppermat[idx])

    u_lower = np.hstack(u_lower)
    u_upper = np.hstack(u_upper)
    # print(u_lower, u_upper)
    scaled_lower, scaled_upper = lin_bound_map(u_lower, u_upper, DinvA)
    y_lower = scaled_lower + Dinvb
    y_upper = scaled_upper + Dinvb
    y_lowermat = iter_to_lower_bound_map[y]
    y_uppermat = iter_to_upper_bound_map[y]
    y_lowermat[k] = y_lower.reshape(-1, )
    y_uppermat[k] = y_upper.reshape(-1, )
    # print(y_lowermat, y_uppermat)


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


def map_linstep_to_iters(y, u, k, iter_to_id_map):
    u_idx = []

    seen_iter = {}
    for x in u:
        if x.is_param:
            u_idx.append(None)
        else:
            if x in seen_iter:
                idx = seen_iter[x] - 1
            else:
                idx = curr_or_prev(y, x, k, iter_to_id_map)
            u_idx.append(idx)
            seen_iter[x] = idx

    return u_idx

def curr_or_prev(var1, var2, k, iter_id_map):
    """
    Returning which step of var2 to use
    I.e. if y = LinStep(x), need to know if y^{k} depends on x^k or x^{k-1}
    """
    i1 = iter_id_map[var1]
    i2 = iter_id_map[var2]
    if i1 <= i2:
        return k-1
    return k
