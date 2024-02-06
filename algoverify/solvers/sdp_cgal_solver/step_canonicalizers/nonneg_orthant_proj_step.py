import numpy as np
import scipy.sparse as spa


def nonneg_orthant_proj_canon(step, k, handler):
    y = step.get_output_var()
    x = step.get_input_var()
    problem_dim = handler.problem_dim

    A_vals = []
    b_lvals = []
    b_uvals = []

    for i in range(handler.num_samples):
        sample_dict = handler.sample_iter_bound_map[i]
        A_curr = []
        b_lcurr = []
        b_ucurr = []
        y_bounds = sample_dict[k][y]
        x_bounds = sample_dict[k][x]
        (y_l, y_u) = y_bounds
        (x_l, x_u) = x_bounds
        x_dim = x.get_dim()
        for j in range(x_dim):
            outmat = spa.lil_matrix((problem_dim, problem_dim))
            outmat[y_l + j, -1] = .5
            outmat[-1, y_l + j] = .5
            b_l = 0
            b_u = np.inf
            A_curr += [outmat]
            b_lcurr += [b_l]
            b_ucurr += [b_u]
            # print('mat', outmat)

        for j in range(x_dim):
            for k in range(x_dim):
                outmat = spa.lil_matrix((problem_dim, problem_dim))
                if j != k:
                    outmat[y_l + j, y_l + k] = .5
                    outmat[y_l + k, y_l + j] = .5
                else:
                    outmat[y_l + j, y_l + k] = 1
                b_l = 0
                b_u = np.inf
                A_curr += [outmat]
                b_lcurr += [b_l]
                b_ucurr += [b_u]
                # print('mat', outmat)

        for j in range(x_dim):
            outmat = spa.lil_matrix((problem_dim, problem_dim))
            outmat[y_l + j, -1] = 1
            outmat[x_l + j, -1] = -1
            outmat = (outmat + outmat.T) / 2
            b_l = 0
            b_u = np.inf
            A_curr += [outmat]
            b_lcurr += [b_l]
            b_ucurr += [b_u]
            # print('mat', outmat)

        for j in range(x_dim):
            outmat = spa.lil_matrix((problem_dim, problem_dim))
            outmat[y_l + j, y_l + j] = 1
            outmat[y_l + j, x_l + j] = -1
            outmat = (outmat + outmat.T) / 2
            b_l = 0
            b_u = 0
            A_curr += [outmat]
            b_lcurr += [b_l]
            b_ucurr += [b_u]
            # print('mat', outmat)

        A_vals += A_curr
        b_lvals += b_lcurr
        b_uvals += b_ucurr

    # print(len(A_vals), len(b_lvals), len(b_uvals))
    return A_vals, b_lvals, b_uvals
