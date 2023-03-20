import numpy as np

# import scipy.sparse as spa


def box_set_preprocess(init_set):
    l = init_set.l
    u = init_set.u
    l = l.reshape((-1, 1))
    u = u.reshape((-1, 1))
    n = l.shape[0]
    b_l = -np.inf * np.ones((n, 1))
    b_u = -np.multiply(l, u)
    return n, b_l, b_u


def box_set_primitive_2(u, init_set, z_vals, handler):
    # l = init_set.l
    # u = init_set.u
    # x = init_set.get_iterate()
    # n = x.get_dim()
    # problem_dim = handler.problem_dim
    # # print('here')
    # num_matrices = z_vals.shape[0]
    # assert n == num_matrices, f'box set primitive 2, dim {n} != {num_matrices} z_vals'
    # x_indices = handler.init_iter_range_map[x]
    # x_range = range(x_indices[0], x_indices[1])

    # # out = np.zeros(problem_dim)

    # for i in range(z_vals.shape[0]):
    #     zi = z_vals[i]
    #     xi = x_range[i]
    #     li = l[i]
    #     ui = u[i]
    #     # Ai is the matrix s.t. Ai[xi, xi] = 1, Ai[xi, -1] = -(li+ui)/2, A[-1, xi] = -(li+ui)/2
    #     m = -(li + ui) / 2
    #     # TODO finish, dont need to build Ai since it only has 3 elements anyways
    pass
