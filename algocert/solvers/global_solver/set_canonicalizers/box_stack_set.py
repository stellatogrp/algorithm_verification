# import gurobipy as gp
import numpy as np

# from algocert.init_set.init_set import InitSet
# from algocert.variables.parameter import Parameter


def box_stack_set_canon(init_set, model, var_to_gp_var_map):
    x = init_set.get_iterate()
    x_var = var_to_gp_var_map[x]
    # l_vec = l_new.reshape(-1, )
    # u_vec = u_new.reshape(-1, )
    var_stack = init_set.var_stack
    if x.is_param:
        x_constr = x_var
    else:
        x_constr = x_var[0]
    curr_dim = 0
    for var_set in var_stack:
        if type(var_set) == list:
            l = var_set[0]
            u = var_set[1]
            n = l.shape[0]
            model.addConstr(l.reshape(-1, ) <= x_constr[curr_dim: curr_dim + n])
            model.addConstr(x_constr[curr_dim: curr_dim + n] <= u.reshape(-1, ))
        else:
            var = var_set.get_iterate()
            n = var.get_dim()
            gp_var = var_to_gp_var_map[var]
            model.addConstr(x_constr[curr_dim: curr_dim + n] == gp_var)
        curr_dim += n

    # print(issubclass(type(bset), InitSet))

    # if x.is_param:
    #     # print(l.shape, x_var.shape)
    #     model.addConstr(l_vec <= x_var)
    #     model.addConstr(x_var <= u_vec)
    # else:
    #     # print(l.shape, x_var[0].shape)
    #     model.addConstr(l_vec <= x_var[0])
    #     model.addConstr(x_var[0] <= u_vec)


def box_stack_set_bound_canon(init_set):
    # x = init_set.get_iterate()
    var_stack = init_set.var_stack
    l_new = []
    u_new = []
    for var_set in var_stack:
        if type(var_set) == list:
            l = var_set[0]
            u = var_set[1]
            l_new.append(l)
            u_new.append(u)
        else:
            l = var_set.l
            u = var_set.u
            l_new.append(l)
            u_new.append(u)
    l_ret = np.vstack(l_new)
    u_ret = np.vstack(u_new)
    return l_ret.reshape(-1, ), u_ret.reshape(-1, )
