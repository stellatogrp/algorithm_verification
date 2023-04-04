# import numpy as np
# import scipy.sparse as spa


def nonneg_lin_canon(step, k, handler):
    # problem_dim = handler.problem_dim
    # y = step.get_output_var()
    # n = y.get_dim()
    # u = step.get_input_var()  # remember this is a stack of variables
    # b = step.get_rhs_const_vec()
    # C = step.get_rhs_matrix()
    # C_blocks = step.C_blocks
    # iter_to_id_map = handler.iterate_to_id_map
    # iter_to_k_map = {}
    # block_size = len(C_blocks)
    # assert block_size == len(u)

    A_vals = []
    b_lvals = []
    b_uvals = []

    # u_dim = step.u_dim
    # mats = invert_matrix_blocking(C, u)

    # for x in u:
    #     if not x.is_param:
    #         if iter_to_id_map[y] <= iter_to_id_map[x]:
    #             iter_to_k_map[x] = k-1
    #         else:
    #             iter_to_k_map[x] = k

    # y_dim = y.get_dim()

    # for i in range(handler.num_samples):
    #     sample_dict = handler.sample_iter_bound_map[i]
    #     iter_bounds_map = {}
    #     # print(sample_dict)
    #     A_curr = []
    #     b_lcurr = []
    #     b_ucurr = []
    #     # u_bounds = []
    #     for x in u:
    #         if x.is_param:
    #             bounds = sample_dict[x]
    #         else:
    #             k_val = iter_to_k_map[x]
    #             bounds = sample_dict[k_val][x]
    #         # print(bounds)
    #         iter_bounds_map[x] = bounds

    #     # first, y >= 0
    #     y_bounds = sample_dict[k][y]
    #     (y_l, y_u) = y_bounds
    #     for j in range(n):
    #         outmat = spa.lil_matrix((problem_dim, problem_dim))
    #         yj = y_l + j
    #         outmat[yj, -1] = .5
    #         outmat[-1, yj] = .5
    #         A_curr += [outmat]
    #         b_lcurr += [0]
    #         b_ucurr += [np.inf]

    #     # next, y - Cu >= b
    #     for j in range(n):
    #         outmat = spa.lil_matrix((problem_dim, problem_dim))
    #         bj = b[j, 0]
    #         yj = y_l + j
    #         outmat[yj, -1] = .5
    #         outmat[-1, yj] = .5
    #         # TODO: finish this

    # # exit(0)

    return A_vals, b_lvals, b_uvals


def invert_matrix_blocking(A, u):
    blocks = []
    start = 0
    # print(A.shape)
    # print(A)
    for iterate in u:
        dim = iterate.get_dim()
        submat = A[:, start: start + dim]
        start += dim
        # print(submat.shape)
        blocks.append(submat)
    return blocks
