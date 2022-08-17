import numpy as np
import scipy.sparse as spa


def box_set_canon(init_set, model, var_to_gp_var_map):
    x = init_set.get_iterate()
    l = init_set.l
    u = init_set.u
    x_var = var_to_gp_var_map[x]

    if x.is_param:
        # print(l.shape, x_var.shape)
        model.addConstr(l <= x_var)
        model.addConstr(x_var <= u)
    else:
        # print(l.shape, x_var[0].shape)
        model.addConstr(l <= x_var[0])
        model.addConstr(x_var[0] <= u)
