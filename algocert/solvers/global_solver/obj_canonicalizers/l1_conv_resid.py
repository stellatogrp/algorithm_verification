import numpy as np

# from gurobipy import GRB


def l1_conv_resid_canon(iterate, model, iterate_to_gp_var_map):
    x = iterate.get_iterate()
    n = x.get_dim()

    x_var = iterate_to_gp_var_map[x]
    xK = x_var[-1]
    x_shape = xK.shape
    xKminus1 = x_var[-2]

    t = xK - xKminus1
    # introduce new variables y, z where y = (t)_+ and z = (-t)_-
    # because then |t| = y - z
    y = model.addMVar(x_shape)
    z = model.addMVar(x_shape)
    # z = model.addMVar(x_shape,
    #                   lb = -GRB.INFINITY * np.ones(x_shape),
    #                   ub = np.zeros(x_shape))

    model.addConstr(y >= t)
    ydiff = y - t
    model.addConstr(y @ ydiff == 0)

    model.addConstr(z >= -t)
    zdiff = z + t
    model.addConstr(z @ zdiff == 0)

    return np.ones(n) @ (y + z)  # , t, y, z
