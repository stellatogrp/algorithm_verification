import numpy as np

# import gurobipy as gp
# from algocert.init_set.init_set import InitSet
from algocert.variables.parameter import Parameter


def stack_set_canon(init_set, model, var_to_gp_var_map):
    # x = init_set.get_iterate()
    # x_var = var_to_gp_var_map[x]
    # # l_vec = l_new.reshape(-1, )
    # # u_vec = u_new.reshape(-1, )
    # var_stack = init_set.var_stack
    # if x.is_param:
    #     x_constr = x_var
    # else:
    #     x_constr = x_var[0]
    # curr_dim = 0
    # for var_set in var_stack:
    #     if type(var_set) == list:
    #         l = var_set[0]
    #         u = var_set[1]
    #         n = l.shape[0]
    #         model.addConstr(l.reshape(-1, ) <= x_constr[curr_dim: curr_dim + n])
    #         model.addConstr(x_constr[curr_dim: curr_dim + n] <= u.reshape(-1, ))
    #     else:
    #         var = var_set.get_iterate()
    #         n = var.get_dim()
    #         gp_var = var_to_gp_var_map[var]
    #         model.addConstr(x_constr[curr_dim: curr_dim + n] == gp_var)
    #     curr_dim += n
    x = init_set.get_iterate()
    x_var = var_to_gp_var_map[x]
    var_stack = init_set.stack
    if x.is_param:
        x_constr = x_var
    else:
        x_constr = x_var[0]

    curr_dim = 0
    for curr_var in var_stack:
        if isinstance(curr_var, Parameter):
            n = curr_var.get_dim()
            gp_var = var_to_gp_var_map[curr_var]
            model.addConstr(x_constr[curr_dim: curr_dim + n] == gp_var)
        else:
            l_val = curr_var[0].reshape(-1, )
            u_val = curr_var[1].reshape(-1, )
            n = l_val.shape[0]
            model.addConstr(x_constr[curr_dim: curr_dim + n] <= u_val)
            model.addConstr(l_val <= x_constr[curr_dim: curr_dim + n])

            # model.addConstr(l.reshape(-1, ) <= x_constr[curr_dim: curr_dim + n])
            # model.addConstr(x_constr[curr_dim: curr_dim + n] <= u.reshape(-1, ))
        curr_dim += n


def stack_set_bound_canon(init_set, handler):
    # var_stack = init_set.var_stack
    # l_new = []
    # u_new = []
    # for var_set in var_stack:
    #     if type(var_set) == list:
    #         l = var_set[0]
    #         u = var_set[1]
    #         l_new.append(l)
    #         u_new.append(u)
    #     else:
    #         l = var_set.l
    #         u = var_set.u
    #         l_new.append(l)
    #         u_new.append(u)
    # l_ret = np.vstack(l_new)
    # u_ret = np.vstack(u_new)
    # return l_ret.reshape(-1, ), u_ret.reshape(-1, )
    var_stack = init_set.stack
    print(var_stack)
    l_new = []
    u_new = []
    for x in var_stack:
        if isinstance(x, Parameter):
            l_new.append(handler.param_to_lower_bound_map[x])
            u_new.append(handler.param_to_upper_bound_map[x])
        else:
            l_new.append(x[0])
            u_new.append(x[1])
    l_ret = np.hstack(l_new)
    u_ret = np.hstack(u_new)
    return l_ret.reshape(-1, ), u_ret.reshape(-1, )
