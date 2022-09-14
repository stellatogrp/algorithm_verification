from gurobipy import GRB


def outer_prod_trace_canon(iterate, model, iterate_to_gp_var_map):
    x = iterate.get_iterate()
    n = x.get_dim()

    x_var = iterate_to_gp_var_map[x]
    xN = x_var[-1]

    obj = 0
    for i in range(n):
        obj += xN[i] ** 2

    model.setObjective(obj, GRB.MAXIMIZE)
