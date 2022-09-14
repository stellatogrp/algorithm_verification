def const_set_canon(init_set, model, var_to_gp_var_map):
    x = init_set.get_iterate()
    val = init_set.get_val()
    x_var = var_to_gp_var_map[x]
    val_vec = val.reshape(-1,)

    if x.is_param:
        model.addConstr(x_var == val)
    else:
        model.addConstr(x_var[0] == val_vec)


def const_set_bound_canon(init_set):
    #  x = init_set.get_iterate()
    val = init_set.get_val()
    val_vec = val.reshape(-1, )
    return val_vec, val_vec
