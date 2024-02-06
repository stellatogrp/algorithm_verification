import scipy.sparse as spa


def conv_resid_canon(obj, handler):
    problem_dim = handler.problem_dim
    K = handler.K
    N = handler.num_samples
    sample_iter_bound_map = handler.sample_iter_bound_map
    x = obj.get_iterate()
    x_dim = x.get_dim()
    output_mat = spa.lil_matrix((problem_dim, problem_dim))
    for i in range(N):
        sample_dict = sample_iter_bound_map[i]
        print(sample_dict)
        xK_bound = sample_dict[K][x]
        xKminus1_bound = sample_dict[K - 1][x]
        (xK_l, xK_u) = xK_bound
        (xKminus1_l, xKminus1_u) = xKminus1_bound
        output_mat[xK_l: xK_u, xK_l: xK_u] = spa.eye(x_dim)
        output_mat[xKminus1_l: xKminus1_u, xKminus1_l: xKminus1_u] = spa.eye(x_dim)
        output_mat[xK_l: xK_u, xKminus1_l: xKminus1_u] = -spa.eye(x_dim)
        output_mat[xKminus1_l: xKminus1_u, xK_l: xK_u] = -spa.eye(x_dim)
    output_mat /= N
    return output_mat
