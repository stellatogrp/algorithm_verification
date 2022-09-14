import numpy as np


def vec_span_set_canon(init_set, model, var_to_gp_var_map):
    x = init_set.get_iterate()
    v = init_set.v
    a = init_set.a
    b = init_set.b
    x_var = var_to_gp_var_map[x]
    # cp.quad_form(x_var, Q) - 2 * (c.T @ Q) @ x_var + c.T @ Q @ c <= 1

    c = model.addMVar(1,
                      ub=a * np.ones(1),
                      lb=b * np.ones(1))

    model.addConstr(x_var == c * v)
