def ellipsoidal_set_canon(init_set, model, var_to_gp_var_map):
    x = init_set.get_iterate()
    Q = init_set.Q
    c = init_set.c
    x_var = var_to_gp_var_map[x]
    # cp.quad_form(x_var, Q) - 2 * (c.T @ Q) @ x_var + c.T @ Q @ c <= 1
    if x.is_param:
        model.addConstr(x_var @ Q @ x_var - 2 * (c.T @ Q) @ x_var + c.T @ Q @ c <= 1)
    else:
        model.addConstr(x_var[0] @ Q @ x_var[0] - 2 * (c.T @ Q) @ x_var[0] + c.T @ Q @ c <= 1)
