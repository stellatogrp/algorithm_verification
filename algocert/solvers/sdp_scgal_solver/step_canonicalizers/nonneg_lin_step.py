import numpy as np


def nonneg_lin_preprocess(step):
    y = step.get_output_var()
    n = y.get_dim()
    # u = step.get_input_var()  # remember this is a stack of variables
    b = step.get_rhs_const_vec()
    # C = step.get_rhs_matrix()
    # C_blocks = step.C_blocks
    # block_size = len(C_blocks)
    # assert block_size == len(u)
    b_l = np.zeros((3 * n, 1))
    b_l[n: 2 * n, :] = -b
    b_u = np.inf * np.ones((3 * n, 1))
    b_u[2 * n: 3 * n, :] = 0
    # print(b_l, b_u)
    return 3 * n, b_l, b_u


def nonneg_lin_primitive_2(step, z_vals, handler):
    # y = step.get_output_var()
    # u = step.get_input_var()  # remember this is a stack of variables
    # b = step.get_rhs_const_vec()
    # C = step.get_rhs_matrix()
    # C_blocks = step.C_blocks
    # block_size = len(C_blocks)
    # assert block_size == len(u)
    # TODO: finish
    pass


def curr_or_prev(var1, var2, iter_id_map):
    """
    Returning 0 means to use the previous iteration handler for var2
    Returning 1 means to use the current
    """
    i1 = iter_id_map[var1]
    i2 = iter_id_map[var2]
    if i1 <= i2:
        return 0
    return 1
