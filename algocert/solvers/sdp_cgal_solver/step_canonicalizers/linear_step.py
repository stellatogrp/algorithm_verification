import numpy as np
import scipy.sparse as spa


def check_symmetric(a, tol=1e-6):
    return np.all(np.abs(a-a.T) < tol)


def linear_step_canon(step, k, handler):
    D = step.get_lhs_matrix()
    D = D.todense()
    C = step.get_rhs_matrix()
    C = C.todense()
    b = step.get_rhs_const_vec()
    b = b.reshape(-1, 1)
    y = step.get_output_var()
    u = step.get_input_var()  # remember that this is a block of variables
    # u_dim = step.u_dim
    iter_to_id_map = handler.iterate_to_id_map
    iter_to_k_map = {}

    y_dim = y.get_dim()

    for x in u:
        if not x.is_param:
            if iter_to_id_map[y] <= iter_to_id_map[x]:
                iter_to_k_map[x] = k-1
            else:
                iter_to_k_map[x] = k
    # print(iter_to_k_map)

    mats = invert_matrix_blocking(C, u)
    # print(mats)

    problem_dim = handler.problem_dim
    # output_mat = spa.lil_matrix((problem_dim, problem_dim))

    A_vals = []
    b_lvals = []
    b_uvals = []
    for i in range(handler.num_samples):
        sample_dict = handler.sample_iter_bound_map[i]
        iter_bounds_map = {}
        # print(sample_dict)
        A_curr = []
        b_lcurr = []
        b_ucurr = []
        # u_bounds = []
        for x in u:
            if x.is_param:
                bounds = sample_dict[x]
            else:
                k_val = iter_to_k_map[x]
                bounds = sample_dict[k_val][x]
            # print(bounds)
            iter_bounds_map[x] = bounds

        # first, Dy = Cu + b
        for j in range(y_dim):
            outmat = spa.lil_matrix((problem_dim, problem_dim))
            y_bounds = sample_dict[k][y]
            (y_l, y_u) = y_bounds
            outmat[y_l: y_u, -1] = D[j]
            # outmat[-1, y_l: y_u] = D[j]
            start = 0
            for x in u:
                if x.is_param:
                    x_bounds = sample_dict[x]
                else:
                    x_bounds = sample_dict[iter_to_k_map[x]][x]
                # print(x_bounds)
                x_l, x_u = x_bounds
                outmat[x_l: x_u, -1] = -C[j, start: start + x_u - x_l]
                # outmat[-1, x_l: x_u] = -C[j, start: start + x_u - x_l]
                start = x_u - x_l
            outmat = (outmat + outmat.T) / 2
            # print((outmat.todense() == outmat.todense().T))
            # print(check_symmetric(outmat.todense()))
            b_l = b[j, 0]
            b_u = b[j, 0]
            A_curr += [outmat]
            b_lcurr += [b_l]
            b_ucurr += [b_u]

        # next, D yyT DT - C uuT CT - C u bT - b uT CT = bbT
        for m in range(y_dim):
            for n in range(y_dim):
                outmat = spa.lil_matrix((problem_dim, problem_dim))
                y_bounds = sample_dict[k][y]
                (y_l, y_u) = y_bounds
                # first D yyT DT
                Dm = D[m]
                Dn = D[n]
                DmDnT = np.outer(Dm, Dn)
                outmat[y_l: y_u, y_l: y_u] = (DmDnT + DmDnT.T) / 2

                # next C uuT CT
                for j in range(len(u)):
                    for k in range(j, len(u)):
                        x = u[j]
                        z = u[k]
                        # x_dim = x.get_dim()
                        # z_dim = z.get_dim()
                        A = mats[j]
                        B = mats[k]
                        (x_l, x_u) = iter_bounds_map[x]
                        (z_l, z_u) = iter_bounds_map[z]
                        # print(x, iter_bounds_map[x], z, iter_bounds_map[z])
                        # print(A.shape, B.shape)
                        # print(A[i], B[j])
                        # print(np.outer(A[i], B[:, j]).shape)
                        abT = -np.outer(A[m], B[n])
                        if j == k:
                            outmat[x_l: x_u, z_l: z_u] = abT
                        else:
                            outmat[x_l: x_u, z_l: z_u] = abT
                            outmat[z_l: z_u, x_l: x_u] = abT.T
                            # outmat[z_l: z_u, x_l: x_u] = abT.T / 2
                        # if j != k:
                        #     outmat[z_l: z_u, x_l: x_u] = abT.T / 2
                # then C u bT and b uT CT
                for j in range(len(u)):
                    x = u[j]
                    A = mats[j]
                    (x_l, x_u) = iter_bounds_map[x]
                    abT = -np.outer(A[m], b[n])
                    outmat[x_l: x_u, -1] = abT / 2
                    outmat[-1, x_l: x_u] = abT / 2

                # print('mat', outmat)
                # print((outmat.todense() == outmat.todense().T))
                outmat = (outmat + outmat.T)/2
                # print(check_symmetric(outmat.todense()))
                b_l = b[m, 0] * b[n, 0]
                b_u = b[m, 0] * b[n, 0]
                A_curr += [outmat]
                b_lcurr += [b_l]
                b_ucurr += [b_u]

        # lastly, DyuT - C uuT - buT = 0

        # TODO: finish
        # I_udim = spa.csc_matrix(spa.eye(u_dim))

        # for n in range(u_dim):
        #     print(step.map_overall_dim_to_x(n))

        # for m in range(y_dim):
        #     # overall_udim = 0
        #     outmat = spa.lil_matrix((problem_dim, problem_dim))
        #     y_bounds = sample_dict[k][y]
        #     (y_l, y_u) = y_bounds
        #     Dm = D[m]
        #     # for j in range(len(u)):
        #     #     x = u[j]
        #     for n in range(u_dim):
        #         In = I_udim[n]
        #         DIT = np.outer(Dm, In)
        #         curr_udim = 0
        #         for j in range(len(u)):
        #             # first, DyuT - buT
        #             x = u[j]
        #             x_dim = x.get_dim()
        #             (x_l, x_u) = iter_bounds_map[x]

        #             # for k in range(j, len(u)):
        #             #     z = u[k]
        #             #     A = mats[j]
        #             #     B = mats[k]
        #             #     (x_l, x_u) = iter_bounds_map[x]
        #             #     (z_l, z_u) = iter_bounds_map[z]

        # TODO: finish
        for m in range(y_dim):
            Dm = D[m]
            # bi = b[i, 0]
            y_bounds = sample_dict[k][y]
            (y_l, y_u) = y_bounds
            overall_udim = 0
            # for n in range(u_dim):
            #     outmat = spa.lil_matrix((problem_dim, problem_dim))
            for j in range(len(u)):
                x = u[j]
                x_dim = x.get_dim()
                (x_l, x_u) = iter_bounds_map[x]
                for l in range(x_dim):
                    # first, D yuT
                    outmat = spa.lil_matrix((problem_dim, problem_dim))
                    outmat[y_l: y_u, l+overall_udim] = Dm.T
                    # print(outmat[y_l: y_u, l+overall_udim].shape, Dm.T.shape)

                    # next, -buT
                    # outmat[]

                overall_udim += x_dim

        # exit(0)

        A_vals += A_curr
        b_lvals += b_lcurr
        b_uvals += b_ucurr

    # print(len(A_vals), len(b_lvals), len(b_uvals))
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


def single_constraint(left, right, bounds, problem_dim, i, j):
    pass
