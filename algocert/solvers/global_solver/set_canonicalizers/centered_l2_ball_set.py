import scipy.sparse as spa


def centered_l2_ball_canon(init_set, model, var_to_gp_var_map):
    x = init_set.get_iterate()
    n = x.get_dim()
    r = init_set.r
    # In = spa.eye(n)
    In = spa.eye(n)
    x_var = var_to_gp_var_map[x]

    if x.is_param:
        model.addConstr(x_var @ In @ x_var <= r ** 2)
    else:
        model.addConstr(x_var[0] @ In @ x_var[0] <= r ** 2)
