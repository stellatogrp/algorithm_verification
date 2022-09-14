from gurobipy import GRB


def linf_conv_resid_canon(iterate, model, iterate_to_gp_var_map):
    x = iterate.get_iterate()
    n = x.get_dim()

    x_var = iterate_to_gp_var_map[x]
    xN = x_var[-1]
    shape = xN.shape
    xNminus1 = x_var[-2]

    # obj = 0
    # for i in range(n):
    #     obj += xN[i] ** 2
    M = 100
    t = model.addVar()
    z_pos = model.addMVar(shape, vtype=GRB.BINARY)
    z_neg = model.addMVar(shape, vtype=GRB.BINARY)
    for i in range(n):
        model.addConstr((xN[i] - xNminus1[i]) <= t)
        model.addConstr(t <= (xN[i] - xNminus1[i]) + M * (1-z_pos[i]))
        model.addConstr(-(xN[i] - xNminus1[i]) <= t)
        model.addConstr(t <= -(xN[i] - xNminus1[i]) + M * (1-z_neg[i]))
    model.addConstr(z_pos.sum() + z_neg.sum() == 1)

    model.setObjective(t, GRB.MAXIMIZE)
