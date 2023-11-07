import gurobipy as gp
import numpy as np

from algocert.solvers.global_solver.step_canonicalizers.linear_step import map_linstep_to_iters
from algocert.variables.parameter import Parameter


def linear_max_proj_canon(step, model, k, iter_to_gp_var_map, param_to_gp_var_map, iter_to_id_map):
    step_data = step.get_matrix_data(k)
    A = step_data['A']
    b = step_data['b']

    y = step.get_output_var()
    u = step.get_input_var()  # remember that this is a block of variables
    y.get_dim()

    l = step.get_lower_bound_vec()
    if not isinstance(l, Parameter):
        l_vec = l.reshape(-1, )
    else:
        l_vec = param_to_gp_var_map[l]

    y_var = iter_to_gp_var_map[y][k]

    A_blocks = step.split_matrix(A)

    w = model.addMVar(y_var.shape,
                      ub=gp.GRB.INFINITY * np.ones(y_var.shape),
                      lb=-gp.GRB.INFINITY * np.ones(y_var.shape))

    w_rhs = b.reshape(-1, )

    u_idx = map_linstep_to_iters(y, u, k, iter_to_id_map)

    for i, x in enumerate(u):
        idx = u_idx[i]
        if x.is_param:
            # print(x)
            x_var = param_to_gp_var_map[x]
        else:
            x_varmatrix = iter_to_gp_var_map[x]
            if iter_to_id_map[y] <= iter_to_id_map[x]:
                x_var = x_varmatrix[k-1]
            else:
                x_var = x_varmatrix[k]
            # x_var = x_varmatrix[idx]
        w_rhs += A_blocks[i] @ x_var

    model.addConstr(w == w_rhs)

    proj_indices = np.array(step.proj_indices)
    nonproj_indices = np.array(step.nonproj_indices)

    # zy = model.addMVar(y_var.shape,
    #                    ub=gp.GRB.INFINITY * np.ones(y_var.shape),
    #                    lb=-gp.GRB.INFINITY * np.ones(y_var.shape))

    # zl = model.addMVar(y_var.shape,
    #                    ub=gp.GRB.INFINITY * np.ones(y_var.shape),
    #                    lb=-gp.GRB.INFINITY * np.ones(y_var.shape))

    # model.addConstr(zy == y_var - w)
    # model.addConstr(zl == y_var - l_vec)

    for idx in nonproj_indices:
        model.addConstr(y_var[idx] == w[idx])

    for idx in proj_indices:
        # model.addConstr(zy[idx] * zl[idx] == 0)
        model.addConstr((y_var[idx] - w[idx]) * (y_var[idx] - l_vec[idx]) == 0)


def linear_max_proj_bound_canon(step, k, iter_to_id_map,
                                iter_to_lower_bound_map, iter_to_upper_bound_map,
                                param_to_lower_bound_map, param_to_upper_bound_map):

    u = step.get_input_var()  # remember this is a list of vars
    y = step.get_output_var()
    l = step.get_lower_bound_vec()
    # nonneg_indices = step.nonneg_indices
    proj_indices = np.array(step.proj_indices)
    np.array(step.nonproj_indices)
    # print(proj_indices, nonproj_indices)

    step_data = step.get_matrix_data(k)
    A = step_data['A']
    step_data['b']

    if not isinstance(l, Parameter):
        l_vec = l.reshape(-1, )
        lower_l = l_vec
        upper_l = l_vec
    else:
        # l_vec = param_to_gp_var_map[l]
        lower_l = param_to_lower_bound_map[l]
        upper_l = param_to_upper_bound_map[l]

    # print(lower_l, upper_l)
    u_lower = []
    u_upper = []
    map_linstep_to_iters(y, u, k, iter_to_id_map)

    for x in u:
        if x.is_param:
            u_lower.append(param_to_lower_bound_map[x])
            u_upper.append(param_to_upper_bound_map[x])
            # print(param_to_lower_bound_map[x].shape)
        else:
            x_lowermat = iter_to_lower_bound_map[x]
            x_uppermat = iter_to_upper_bound_map[x]
            if iter_to_id_map[y] <= iter_to_id_map[x]:
                x_lower = x_lowermat[k - 1]
                x_upper = x_uppermat[k - 1]
            else:
                x_lower = x_lowermat[k]
                x_upper = x_uppermat[k]
            u_lower.append(x_lower)
            u_upper.append(x_upper)

    # for idx, x in zip(u_idx, u):
    #     if idx is None:
    #         u_lower.append(param_to_lower_bound_map[x])
    #         u_upper.append(param_to_upper_bound_map[x])
    #     else:
    #         x_lowermat = iter_to_lower_bound_map[x]
    #         x_uppermat = iter_to_upper_bound_map[x]
    #         u_lower.append(x_lowermat[idx])
    #         u_upper.append(x_uppermat[idx])

    u_lower = np.hstack(u_lower)
    u_upper = np.hstack(u_upper)

    # print(u_lower, u_upper)
    scaled_lower, scaled_upper = lin_bound_map(u_lower, u_upper, A)
    # print(scaled_lower, scaled_upper)

    y_lower = scaled_lower.copy().reshape(-1, )
    y_upper = scaled_upper.copy().reshape(-1, )

    if len(proj_indices) > 0:
        y_lower[proj_indices] = np.maximum(y_lower[proj_indices], lower_l[proj_indices])
        y_upper[proj_indices] = np.maximum(y_upper[proj_indices], upper_l[proj_indices])

    print(y_lower, y_upper)
    # exit(0)

    y_lowermat = iter_to_lower_bound_map[y]
    y_uppermat = iter_to_upper_bound_map[y]
    y_lowermat[k] = y_lower
    y_uppermat[k] = y_upper

    # print(y_lowermat, y_uppermat)


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
