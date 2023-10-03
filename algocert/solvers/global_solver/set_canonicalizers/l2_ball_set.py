import scipy.sparse as spa


def l2_ball_set_canon(init_set, model, var_to_gp_var_map):
    x = init_set.get_iterate()
    n = x.get_dim()
    r = init_set.r
    c = init_set.c
    # In = spa.eye(n)
    In = spa.eye(n)
    x_var = var_to_gp_var_map[x]

    if x.is_param:
        model.addConstr(x_var @ In @ x_var - 2 * c.T @ x_var + c.T @ c <= r ** 2)
    else:
        model.addConstr(x_var[0] @ In @ x_var[0] - 2 * c.T @ x_var[0] + c.T @ c <= r ** 2)

    # c = init_set.c
    # x_var = var_to_gp_var_map[x]
    # # cp.quad_form(x_var, Q) - 2 * (c.T @ Q) @ x_var + c.T @ Q @ c <= 1
    # if x.is_param:
    #     model.addConstr(x_var @ Q @ x_var - 2 * (c.T @ Q) @ x_var + c.T @ Q @ c <= 1)
    # else:
    #     model.addConstr(x_var[0] @ Q @ x_var[0] - 2 * (c.T @ Q) @ x_var[0] + c.T @ Q @ c <= 1)


def l2_ball_set_bound_canon(init_set):
    r = init_set.r
    c = init_set.c
    u = c + r
    l = c - r
    return l.reshape(-1, ), u.reshape(-1, )
