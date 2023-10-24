import gurobipy as gp
import numpy as np
import scipy.sparse as spa


def block_conv_resid_canon(obj, model, iterate_to_gp_var_map):
    s = obj.get_iterate()
    A = obj.get_block_mat()
    n = A.shape[0]
    A_split = obj.split_matrix(A.todense())
    # print(A)

    sk = map_stack_to_gp_var(s, A_split, n, model, iterate_to_gp_var_map, -1)
    skminus1 = map_stack_to_gp_var(s, A_split, n, model, iterate_to_gp_var_map, -2)

    In = spa.eye(n)
    twoIn = 2 * In

    obj = sk @ In @ sk - sk @ twoIn @ skminus1 + skminus1 @ In @ skminus1
    return obj


def map_stack_to_gp_var(s, A, n, model, iterate_to_gp_var_map, k):
    # print(A)
    sk = model.addMVar(n,
                       ub=gp.GRB.INFINITY * np.ones(n),
                       lb=-gp.GRB.INFINITY * np.ones(n))

    sk_val = 0
    for Ai, x in zip(A, s):
        x_var = iterate_to_gp_var_map[x][k]
        sk_val += Ai @ x_var

    model.addConstr(sk == sk_val)
    return sk
