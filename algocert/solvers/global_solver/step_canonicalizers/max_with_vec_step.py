import gurobipy as gp
import numpy as np


def max_vec_canon(step, model, k, iter_to_gp_var_map, param_to_gp_var_map, iter_to_id_map):
    y = step.get_output_var()
    x = step.get_input_var()
    l = step.get_lower_bound_vec()
    l_vec = l.reshape(-1, )

    y_varmatrix = iter_to_gp_var_map[y]
    x_varmatrix = iter_to_gp_var_map[x]

    y_var = y_varmatrix[k]
    # print(iter_to_id_map[y], iter_to_id_map[x])
    if iter_to_id_map[y] <= iter_to_id_map[x]:
        x_var = x_varmatrix[k - 1]
    else:
        x_var = x_varmatrix[k]

    # print(y_var.shape, l.shape)
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