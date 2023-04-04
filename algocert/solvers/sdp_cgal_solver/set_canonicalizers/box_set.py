import numpy as np
import scipy.sparse as spa


def box_set_canon(init_set, handler):
    # print('here')
    x = init_set.get_iterate()
    l = init_set.l
    u = init_set.u
    N = handler.num_samples
    problem_dim = handler.problem_dim
    sample_iter_bound_map = handler.sample_iter_bound_map
    A_vals = []
    b_lvals = []
    b_uvals = []

    if x.is_param:
        for i in range(N):
            # x_dim = x.get_dim()
            # output_mat = spa.lil_matrix((problem_dim, problem_dim))
            sample_dict = sample_iter_bound_map[i]
            x_bound = sample_dict[x]
            (x_l, x_u) = x_bound
            for j in range(x_l, x_u):
                output_mat = spa.lil_matrix((problem_dim, problem_dim))
                # print(j)
                output_mat[j, j] = 1
                l_val = l[j - x_l, 0]
                u_val = u[j - x_l, 0]
                middle = -(l_val + u_val) / 2
                output_mat[j, -1] = middle
                output_mat[-1, j] = middle
                A_vals += [output_mat]
                b_lvals += [-np.inf]
                b_uvals += [-l_val * u_val]
    else:  # x is not a parameter
        # x_dim = x.get_dim()
        init_dict = handler.init_iter_range_map
        # x0_bound = init_dict[x]
        x_bound = init_dict[x]
        (x_l, x_u) = x_bound
        for j in range(x_l, x_u):
            # print(j)
            output_mat = spa.lil_matrix((problem_dim, problem_dim))
            output_mat[j, j] = 1
            l_val = l[j - x_l, 0]
            u_val = u[j - x_l, 0]
            middle = -(l_val + u_val) / 2
            output_mat[j, -1] = middle
            output_mat[-1, j] = middle
            A_vals += [output_mat]
            b_lvals += [-np.inf]
            b_uvals += [-l_val * u_val]
    # print([A.shape for A in A_vals])
    # exit(0)

    return A_vals, b_lvals, b_uvals

    #     if x.is_param:
    #         for i in range(N):
    #             x_dim = x.get_dim()
    #             output_mat = spa.lil_matrix((problem_dim, problem_dim))
    #             sample_dict = sample_iter_bound_map[i]
    #             x_bound = sample_dict[x]
    #             (x_l, x_u) = x_bound
    #             output_mat[x_l: x_u, x_l: x_u] = spa.eye(x_dim)
    #             b_l = 0
    #             b_u = r ** 2
    #             A_vals += [output_mat]
    #             b_lvals += [b_l]
    #             b_uvals += [b_u]

    #     else:
    #         x_dim = x.get_dim()
    #         output_mat = spa.lil_matrix((problem_dim, problem_dim))
    #         init_dict = handler.init_iter_range_map
    #         x0_bound = init_dict[x]
    #         (x0_l, x0_u) = x0_bound
    #         output_mat[x0_l: x0_u, x0_l: x0_u] = spa.eye(x_dim)
    #         b_l = 0
    #         b_u = r ** 2
    #         A_vals += [output_mat]
    #         b_lvals += [b_l]
    #         b_uvals += [b_u]

    #     return A_vals, b_lvals, b_uvals
    pass
    # return [], [], []
