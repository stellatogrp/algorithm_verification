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
