import gurobipy as gp
import numpy as np

from algocert.variables.parameter import Parameter


def max_vec_canon(step, model, k, iter_to_gp_var_map, param_to_gp_var_map, iter_to_id_map):
    y = step.get_output_var()
    x = step.get_input_var()

    y_varmatrix = iter_to_gp_var_map[y]
    x_varmatrix = iter_to_gp_var_map[x]

    y_var = y_varmatrix[k]
    # print(iter_to_id_map[y], iter_to_id_map[x])
    if iter_to_id_map[y] <= iter_to_id_map[x]:
        x_var = x_varmatrix[k - 1]
    else:
        x_var = x_varmatrix[k]

    l = step.get_lower_bound_vec()
    if not type(l) == Parameter:
        l_vec = l.reshape(-1, )
    else:
        l_vec = param_to_gp_var_map[l]

    model.addConstr(y_var >= l_vec)
    model.addConstr(y_var >= x_var)
    # print(y_var.shape)

    w = model.addMVar(y_var.shape,
                      ub=gp.GRB.INFINITY * np.ones(y_var.shape),
                      lb=-gp.GRB.INFINITY * np.ones(y_var.shape))
    z = model.addMVar(y_var.shape,
                      ub=gp.GRB.INFINITY * np.ones(y_var.shape),
                      lb=-gp.GRB.INFINITY * np.ones(y_var.shape))

    model.addConstr(z == y_var - x_var)
    model.addConstr(w == y_var - l_vec)
    model.addConstr(w @ z == 0)


def max_vec_bound_canon(step, k, iter_to_id_map,
                        iter_to_lower_bound_map, iter_to_upper_bound_map,
                        param_to_lower_bound_map, param_to_upper_bound_map):
    y = step.get_output_var()
    x = step.get_input_var()
    l = step.get_lower_bound_vec()

    if not type(l) == Parameter:
        l_vec = l.reshape(-1, )
        lower_l = l_vec
        upper_l = l_vec
    else:
        # l_vec = param_to_gp_var_map[l]
        lower_l = param_to_lower_bound_map[l]
        upper_l = param_to_upper_bound_map[l]

    x_lowermat = iter_to_lower_bound_map[x]
    x_uppermat = iter_to_upper_bound_map[x]
    if iter_to_id_map[y] <= iter_to_id_map[x]:
        x_lower = x_lowermat[k - 1]
        x_upper = x_uppermat[k - 1]
    else:
        x_lower = x_lowermat[k]
        x_upper = x_uppermat[k]
    # print(x_lower, x_upper)
    y_lower = np.maximum(x_lower, lower_l)
    y_upper = np.maximum(x_upper, upper_l)
    y_lowermat = iter_to_lower_bound_map[y]
    y_uppermat = iter_to_upper_bound_map[y]
    y_lowermat[k] = y_lower.reshape(-1, )
    y_uppermat[k] = y_upper.reshape(-1, )
