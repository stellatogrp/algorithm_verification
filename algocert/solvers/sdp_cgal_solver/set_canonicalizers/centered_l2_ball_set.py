import scipy.sparse as spa


def centered_l2_ball_canon(init_set, handler):
    x = init_set.get_iterate()
    r = init_set.r
    N = handler.num_samples
    problem_dim = handler.problem_dim
    sample_iter_bound_map = handler.sample_iter_bound_map
    A_vals = []
    b_lvals = []
    b_uvals = []

    if x.is_param:
        for i in range(N):
            x_dim = x.get_dim()
            output_mat = spa.lil_matrix((problem_dim, problem_dim))
            sample_dict = sample_iter_bound_map[i]
            x_bound = sample_dict[x]
            (x_l, x_u) = x_bound
            output_mat[x_l: x_u, x_l: x_u] = spa.eye(x_dim)
            b_l = 0
            b_u = r ** 2
            A_vals += [output_mat]
            b_lvals += [b_l]
            b_uvals += [b_u]

    else:
        x_dim = x.get_dim()
        output_mat = spa.lil_matrix((problem_dim, problem_dim))
        init_dict = handler.init_iter_range_map
        x0_bound = init_dict[x]
        # print(x0_bound)
        (x0_l, x0_u) = x0_bound
        output_mat[x0_l: x0_u, x0_l: x0_u] = spa.eye(x_dim)
        b_l = 0
        b_u = r ** 2
        A_vals += [output_mat]
        b_lvals += [b_l]
        b_uvals += [b_u]

    return A_vals, b_lvals, b_uvals
