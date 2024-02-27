import numpy as np


def control_example_set_canon(init_set, model, var_to_gp_var_map):
    x = init_set.get_iterate()
    l = init_set.l
    u = init_set.u
    nonzero_n = l.shape[0]
    full_n = x.get_dim()

    zeros = np.zeros((full_n - nonzero_n, 1))
    # l_new = np.concatenate([l, zeros])
    # u_new = np.concatenate([u, zeros])
    l_new = np.vstack([l, zeros])
    u_new = np.vstack([u, zeros])

    x_var = var_to_gp_var_map[x]
    l_vec = l_new.reshape(-1, )
    u_vec = u_new.reshape(-1, )

    if x.is_param:
        # print(l.shape, x_var.shape)
        model.addConstr(l_vec <= x_var)
        model.addConstr(x_var <= u_vec)
    else:
        # print(l.shape, x_var[0].shape)
        model.addConstr(l_vec <= x_var[0])
        model.addConstr(x_var[0] <= u_vec)


def control_example_set_bound_canon(init_set):
    x = init_set.get_iterate()
    l = init_set.l
    u = init_set.u
    nonzero_n = l.shape[0]
    full_n = x.get_dim()
    zeros = np.zeros((full_n - nonzero_n, 1))
    # l_new = np.concatenate([l, zeros])
    # u_new = np.concatenate([u, zeros])
    l_new = np.vstack([l, zeros])
    u_new = np.vstack([u, zeros])
    return l_new.reshape(-1, ), u_new.reshape(-1, )
