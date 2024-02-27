def affine_vec_set_canon(init_set, model, var_to_gp_var_map):
    q = init_set.get_iterate()
    S = init_set.S
    b = init_set.b.reshape(-1, )
    q_var = var_to_gp_var_map[q]
    theta_set = init_set.theta_set
    theta_param = theta_set.get_iterate()

    # if theta_param not in var_to_gp_var_map:
    #     bound_canon_method = BOUND_SET_CANON_METHODS[type(theta_set)]
    #     set_canon_method = SET_CANON_METHODS[type(theta_set)]
    #     l, u = bound_canon_method(theta_set)
    #     theta = model.addMVar(theta_dim,
    #                           ub=u,
    #                           lb=l)
    #     var_to_gp_var_map[theta_param] = theta
    #     set_canon_method(theta_set, model, var_to_gp_var_map)
    theta_var = var_to_gp_var_map[theta_param]
    model.addConstr(q_var == S @ theta_var + b)


def affine_vec_set_bound_canon(init_set):
    pass
