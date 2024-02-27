# import scipy.sparse as spa

# from gurobipy import GRB


def lin_comb_squared_norm_canon(obj, model, iterate_to_gp_var_map):
    x_stack = obj.get_iterate_stack()
    A_stack = obj.get_matrix_stack()

    obj_vec = 0
    for i in range(len(x_stack)):
        x = x_stack[i]
        A = A_stack[i]
        x_var = iterate_to_gp_var_map[x]
        xN = x_var[-1]
        # print(A, xN)
        obj_vec += A @ xN

    n = A.shape[0]
    # In = spa.eye(n)
    # n = x.get_dim()
    #
    # x_var = iterate_to_gp_var_map[x]
    # xN = x_var[-1]
    # xNminus1 = x_var[-2]
    #
    # In = spa.eye(n)
    # twoIn = 2 * In
    #
    # obj = xN @ In @ xN - xN @ twoIn @ xNminus1 + xNminus1 @ In @ xNminus1
    # obj = xN @ In @ xN - xNminus1 @ In @ xNminus1
    # c = np.ones(n)
    # obj = c @ xN
    # model.setObjective(obj, GRB.MAXIMIZE)
    obj = 0
    for i in range(n):
        obj += xN[i] ** 2
    return obj
