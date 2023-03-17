import gurobipy as gp
import numpy as np


def nonneg_lin_canon(step, model, k, iter_to_gp_var_map, param_to_gp_var_map, iter_to_id_map):
    y = step.get_output_var()
    u = step.get_input_var()  # remember this is a stack of variables
    b = step.get_rhs_const_vec()
    C_blocks = step.C_blocks
    constraint_rhs = 0

    y_varmatrix = iter_to_gp_var_map[y]
    y_var = y_varmatrix[k]

    # now create the linstep on the rhs
    for i, x in enumerate(u):
        C = C_blocks[i]
        if x.is_param:
            x_var = param_to_gp_var_map[x]
        else:
            x_varmatrix = iter_to_gp_var_map[x]
            # print(iter_to_id_map[y], iter_to_id_map[x])
            if iter_to_id_map[y] <= iter_to_id_map[x]:
                x_var = x_varmatrix[k-1]
            else:
                x_var = x_varmatrix[k]
        constraint_rhs += C.tocsc() @ x_var
    # print(constraint_rhs.shape, b.shape)
    constraint_rhs += b.reshape(-1, )

    model.addConstr(y_var >= 0)
    model.addConstr(y_var >= constraint_rhs)
    z = model.addMVar(y_var.shape,
                      ub=gp.GRB.INFINITY * np.ones(y_var.shape),
                      lb=-gp.GRB.INFINITY * np.ones(y_var.shape))

    model.addConstr(z == y_var - constraint_rhs)
    model.addConstr(y_var @ z == 0)


def nonneg_lin_bound_canon(step, k, iter_to_id_map,
                           iter_to_lower_bound_map, iter_to_upper_bound_map,
                           param_to_lower_bound_map, param_to_upper_bound_map):

    u = step.get_input_var()  # remember this is a list of vars
    y = step.get_output_var()
    b = step.get_rhs_const_vec()
    # C_blocks = step.C_blocks
    n = y.get_dim()
    C = step.get_rhs_matrix()
    l = np.zeros((n, 1))

    u_lower = []
    u_upper = []

    for x in u:
        if x.is_param:
            u_lower.append(param_to_lower_bound_map[x])
            u_upper.append(param_to_upper_bound_map[x])
        else:
            x_lowermat = iter_to_lower_bound_map[x]
            x_uppermat = iter_to_upper_bound_map[x]
            if iter_to_id_map[y] <= iter_to_id_map[x]:
                x_lower = x_lowermat[k - 1]
                x_upper = x_uppermat[k - 1]
            else:
                x_lower = x_lowermat[k]
                x_upper = x_uppermat[k]
            u_lower.append(x_lower)
            u_upper.append(x_upper)
    u_lower = np.hstack(u_lower)
    u_upper = np.hstack(u_upper)
    # print(u_lower, u_upper)

    scaled_lower, scaled_upper = lin_bound_map(u_lower, u_upper, C)
    # y_lower = scaled_lower + Dinvb
    # y_upper = scaled_upper + Dinvb
    y_lower = np.maximum(scaled_lower + b, l)
    y_upper = np.maximum(scaled_upper + b, l)
    y_lowermat = iter_to_lower_bound_map[y]
    y_uppermat = iter_to_upper_bound_map[y]
    y_lowermat[k] = y_lower.reshape(-1, )
    y_uppermat[k] = y_upper.reshape(-1, )


def lin_bound_map(l, u, A):
    A = A.toarray()
    (m, n) = A.shape
    l_out = np.zeros(m)
    u_out = np.zeros(m)
    for i in range(m):
        lower = 0
        upper = 0
        for j in range(n):
            if A[i][j] >= 0:
                lower += A[i][j] * l[j]
                upper += A[i][j] * u[j]
            else:
                lower += A[i][j] * u[j]
                upper += A[i][j] * l[j]
        l_out[i] = lower
        u_out[i] = upper

    return np.reshape(l_out, (m, 1)), np.reshape(u_out, (m, 1))
