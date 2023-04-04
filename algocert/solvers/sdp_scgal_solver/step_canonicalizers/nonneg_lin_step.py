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


def nonneg_lin_primitive_2(u, step, k, z_vals, sample_iter_bounds, handler):
    problem_dim = handler.problem_dim
    y = step.get_output_var()
    n = y.get_dim()
    x = step.get_input_var()  # remember this is a stack of variables
    b = step.get_rhs_const_vec()
    # C = step.get_rhs_matrix()
    C_blocks = step.C_blocks
    block_size = len(C_blocks)
    assert block_size == len(x)

    out = np.zeros(problem_dim)
    prev = sample_iter_bounds[k-1]
    curr = sample_iter_bounds[k]

    x_ranges = []
    for w in x:
        if w.is_param:
            x_ranges.append(sample_iter_bounds[w])
        elif curr_or_prev(y, w, handler.iterate_to_id_map) == 0:
            x_ranges.append(prev[w])
        else:
            x_ranges.append(curr[w])

    y_indices = curr[y]
    y_range = range(y_indices[0], y_indices[1])

    print(z_vals)
    print(sample_iter_bounds, prev, curr)
    # first, the y >= 0 constraints
    for i in range(n):
        zi = z_vals[i]
        yi = y_range[i]
        print(zi, yi)
        # for this part, we have A[yi, -1] = A[-1. yi] = .5
        if np.abs(zi) >= 1e-7:
            out[yi] += zi * u[-1] * .5
            out[-1] += zi * u[yi] * .5

    # next, the y - Cx >= b constraints
    for i in range(n):
        zk = z_vals[i + n]
        yi = y_range[i]
        print(zk, yi)
        # rows = [yi, problem_dim-1]
        # cols = [problem_dim-1, yi]
        # data = [.5, .5]
        # for (j, w) in enumerate(x):
        #     w_range = x_ranges[j]
        #     Cj = C_blocks[j]
        #     Cj_irow = list(Cj[i].todense()[0])
        #     # print(Cj_irow.shape, Cj_irow)
        #     print(len(Cj_irow), Cj_irow)
        #     print(j, w, w_range)
        out[yi] += zk * u[-1] * .5
        out[-1] += zk * u[yi] * .5
        for (j, w) in enumerate(x):
            # print(j, w)
            w_range = x_ranges[j]
            Cj = C_blocks[j]
            Cj_irow = Cj[i].todense()
            w_offset = w_range[0]
            for k in range(Cj_irow.shape[1]):
                # print(k, k+w_offset)
                # print(Cj_irow[0, k])
                out[k+w_offset] += -.5 * zk * u[-1] * Cj_irow[0, k]
                out[-1] += -.5 * zk * u[k+w_offset] * Cj_irow[0, k]

    # lastly, the diag(M_{yy} - M_{yu}C^T - yb^T) = 0 constraints
    for i in range(n):
        zk = z_vals[i + 2 * n]
        yi = y_range[i]
        bi = b[i, 0]
        print(zk, yi)
        out[yi] += zk * u[yi]
        out[yi] += -.5 * zk * u[-1] * bi
        out[-1] += -.5 * zk * u[yi] * bi
        for (j, w) in enumerate(x):
            # print(j, w)
            w_range = x_ranges[j]
            Cj = C_blocks[j]
            Cj_irow = Cj[i].todense()
            w_offset = w_range[0]
            for k in range(Cj_irow.shape[1]):
                out[yi] += -.5 * zk * u[k+w_offset] * Cj_irow[0, k]
                out[k+w_offset] += -.5 * zk * u[yi] * Cj_irow[0, k]

    return out


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
