import numpy as np
import scipy.sparse as spa


def hl_linear_step_canon(step, model, k, iter_to_gp_var_map, param_to_gp_var_map, iter_to_id_map):
    D = step.get_lhs_matrix()
    A = step.get_rhs_matrix()
    b = step.get_rhs_const_vec()
    y = step.get_output_var()
    u = step.get_input_var()  # remember that this is a block of variables

    y_var = iter_to_gp_var_map[y]
    constraint_lhs = y_var[k]
    constraint_rhs = 0

    # create the boundaries to partition the A matrix
    left = 0
    right = 0
    boundaries = []
    for x in u:
        n = x.get_dim()
        right = left + n
        # print(left, right)
        # print(A.tocsc()[:, left: right].shape)
        boundaries.append((left, right))
        left = right
    # print(boundaries)

    for i, x in enumerate(u):
        (left, right) = boundaries[i]
        if x.is_param:
            # print(x)
            x_var = param_to_gp_var_map[x]
        else:
            x_varmatrix = iter_to_gp_var_map[x]
            # print(iter_to_id_map[y], iter_to_id_map[x])
            if iter_to_id_map[y] <= iter_to_id_map[x]:
                x_var = x_varmatrix[k-1]
            else:
                x_var = x_varmatrix[k]

        constraint_rhs += A.tocsc()[:, left: right] @ x_var
        # print((A.tocsc()[:, left: right] @ x_var).shape)
    # print('rhs', constraint_rhs.shape)
    # print('lhs', constraint_lhs.shape)
    # print(b, b.shape)
    constraint_rhs += b

    model.addConstr(constraint_lhs == constraint_rhs)
