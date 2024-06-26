def box_set_canon(init_set, model, var_to_gp_var_map, k=0):
    x = init_set.get_iterate()
    l = init_set.l
    u = init_set.u
    x_var = var_to_gp_var_map[x]
    l_vec = l.reshape(-1, )
    u_vec = u.reshape(-1, )

    if x.is_param:
        # print(l.shape, x_var.shape)
        model.addConstr(l_vec <= x_var)
        model.addConstr(x_var <= u_vec)
    else:
        # print(l.shape, x_var[0].shape)
        model.addConstr(l_vec <= x_var[k])
        model.addConstr(x_var[k] <= u_vec)


def box_set_bound_canon(init_set, handler):
    #  x = init_set.get_iterate()
    l = init_set.l
    u = init_set.u
    return l.reshape(-1, ), u.reshape(-1, )
