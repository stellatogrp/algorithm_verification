import gurobipy as gp
import numpy as np


def nonneg_orthant_proj_canon(step, model, k, iter_to_gp_var_map, param_to_gp_var_map, iter_to_id_map):
    y = step.get_output_var()
    x = step.get_input_var()

    y_varmatrix = iter_to_gp_var_map[y]
    x_varmatrix = iter_to_gp_var_map[x]

    # model.addConstr(x[k + 1] >= 0)
    # model.addConstr(x[k + 1] >= y[k + 1])
    # model.addConstr(z[k + 1] == x[k + 1] - y[k + 1])
    # model.addConstr(x[k + 1] @ z[k + 1] == 0)

    y_var = y_varmatrix[k]
    # print(iter_to_id_map[y], iter_to_id_map[x])
    if iter_to_id_map[y] <= iter_to_id_map[x]:
        x_var = x_varmatrix[k-1]
    else:
        x_var = x_varmatrix[k]

    model.addConstr(y_var >= 0)
    model.addConstr(y_var >= x_var)
    # print(y_var.shape)

    # z = model.addMVar(y_var.shape,
    #                   ub=gp.GRB.INFINITY * np.ones(y_var.shape),
    #                   lb=-gp.GRB.INFINITY * np.ones(y_var.shape))

    # model.addConstr(z == y_var - x_var)
    # model.addConstr(y_var @ z == 0)

    real_indices = np.array(step.real_indices)
    nonneg_indices = np.array(step.nonneg_indices)

    z_dim = len(nonneg_indices)

    z = model.addMVar(z_dim,
                      ub=gp.GRB.INFINITY * np.ones(z_dim),
                      lb=-gp.GRB.INFINITY * np.ones(z_dim))

    if len(real_indices) > 0:
        model.addConstr(x_var[real_indices] == y_var[real_indices])
    # if len(nonneg_indices) > 0:
    model.addConstr(y_var[nonneg_indices] >= 0)
    model.addConstr(y_var[nonneg_indices] >= x_var[nonneg_indices])

    model.addConstr(z == y_var[nonneg_indices] - x_var[nonneg_indices])
    model.addConstr(y_var[nonneg_indices] @ z == 0)


def nonneg_orthant_proj_bound_canon(step, k, iter_to_id_map,
                                    iter_to_lower_bound_map, iter_to_upper_bound_map,
                                    param_to_lower_bound_map, param_to_upper_bound_map):
    # y = step.get_output_var()
    # x = step.get_input_var()
    # n = x.get_dim()
    # l = np.zeros(n)

    # x_lowermat = iter_to_lower_bound_map[x]
    # x_uppermat = iter_to_upper_bound_map[x]
    # if iter_to_id_map[y] <= iter_to_id_map[x]:
    #     x_lower = x_lowermat[k - 1]
    #     x_upper = x_uppermat[k - 1]
    # else:
    #     x_lower = x_lowermat[k]
    #     x_upper = x_uppermat[k]
    # # print(x_lower, x_upper)
    # # print(x_lower, l)
    # y_lower = np.maximum(x_lower, l)
    # y_upper = np.maximum(x_upper, l)
    # y_lowermat = iter_to_lower_bound_map[y]
    # y_uppermat = iter_to_upper_bound_map[y]
    # y_lowermat[k] = y_lower.reshape(-1, )
    # y_uppermat[k] = y_upper.reshape(-1, )

    y = step.get_output_var()
    x = step.get_input_var()
    # real_indices = np.array(list(step.real_indices))
    # nonneg_indices = np.array(list(step.nonneg_indices))
    nonneg_indices = step.nonneg_indices

    # n = x.get_dim()
    z_dim = len(nonneg_indices)
    l = np.zeros(z_dim)

    x_lowermat = iter_to_lower_bound_map[x]
    x_uppermat = iter_to_upper_bound_map[x]
    if iter_to_id_map[y] <= iter_to_id_map[x]:
        x_lower = x_lowermat[k - 1]
        x_upper = x_uppermat[k - 1]
    else:
        x_lower = x_lowermat[k]
        x_upper = x_uppermat[k]

    y_lower = x_lower.copy()
    y_upper = x_upper.copy()

    y_lower[nonneg_indices] = np.maximum(x_lower[nonneg_indices], l)
    y_upper[nonneg_indices] = np.maximum(x_upper[nonneg_indices], l)

    # print(x_lower, y_lower)
    # print(x_upper, y_upper)

    y_lowermat = iter_to_lower_bound_map[y]
    y_uppermat = iter_to_upper_bound_map[y]
    y_lowermat[k] = y_lower.reshape(-1, )
    y_uppermat[k] = y_upper.reshape(-1, )
