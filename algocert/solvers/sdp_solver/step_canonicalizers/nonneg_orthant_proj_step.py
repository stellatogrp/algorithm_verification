import cvxpy as cp
import numpy as np

from algocert.basic_algorithm_steps.basic_linear_step import BasicLinearStep
from algocert.solvers.sdp_solver.var_bounds.RLT_constraints import \
    RLT_constraints


def nonneg_orthant_proj_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars, add_RLT, kwargs):
    step = steps[i]
    prev_step = steps[i-1]

    y = step.get_output_var()
    x = step.get_input_var()
    n = x.get_dim()

    y_var = curr.iterate_vars[y].get_cp_var()
    yyT_var = curr.iterate_outerproduct_vars[y]
    x_var = curr.iterate_vars[x].get_cp_var()
    xxT_var = curr.iterate_outerproduct_vars[x]

    yxT_var = curr.iterate_cross_vars[y][x]

    constraints = [y_var >= 0, yyT_var >= 0, y_var >= x_var, cp.diag(yyT_var - yxT_var) == 0]
    constraints += [
        cp.bmat([
            [yyT_var, yxT_var, y_var],
            [yxT_var.T, xxT_var, x_var],
            [y_var.T, x_var.T, np.array([[1]])]
        ]) >> 0,
    ]

    if type(prev_step) == BasicLinearStep:
        # print(type(x))
        A = prev_step.get_rhs_matrix()
        D = prev_step.get_lhs_matrix()
        b = prev_step.get_rhs_const_vec()
        block_step = steps[i-2]
        u = block_step.get_output_var()
        u_var = curr.iterate_vars[u].get_cp_var()
        uuT_var = curr.iterate_outerproduct_vars[u]
        block_vars = block_step.list_x
        # print(block_vars)
        yuT_var = curr.iterate_cross_vars[y][u]
        yuT_blocks = []
        for var in block_vars:
            if var.is_param:
                yuT_blocks.append(curr.iterate_param_vars[y][var])
            else:
                yuT_blocks.append(curr.iterate_cross_vars[y][var])
        constraints += [
            yuT_var == cp.bmat([
                yuT_blocks
            ]),
            cp.bmat([
                [yyT_var, yuT_var, y_var],
                [yuT_var.T, uuT_var, u_var],
                [y_var.T, u_var.T, np.array([[1]])]
            ]) >> 0,
            yxT_var @ D.T == yuT_var @ A.T + y_var @ b.T,
        ]

    lower_y = curr.iterate_vars[y].get_lower_bound()
    upper_y = curr.iterate_vars[y].get_upper_bound()
    lower_x = curr.iterate_vars[x].get_lower_bound()
    upper_x = curr.iterate_vars[x].get_upper_bound()

    if add_RLT:
        extra_constraints = RLT_constraints(yxT_var, y_var, lower_y, upper_y, x_var, lower_x, upper_x)
        constraints += extra_constraints

        # constraints += RLT_constraints(xxT_var, x_var, lower_x, upper_x, x_var, lower_x, upper_x)
        # constraints += [lower_x <= x_var, x_var <= upper_x]
        # constraints += RLT_constraints(yyT_var, y_var, lower_y, upper_y, y_var, lower_y, upper_y)
        # constraints += [lower_y <= y_var, y_var <= upper_y]

        if 'add_planet' in kwargs:
            if kwargs['add_planet']:
                test = True
                if test:
                    print((upper_x - lower_x).reshape(-1,))
                    gaps_vec = (upper_x - lower_x).reshape(-1,)
                    pos_gap_indices = np.argwhere(gaps_vec >= 1e-5).reshape(-1, )
                    zero_gap_indices = np.argwhere(gaps_vec < 1e-5).reshape(-1, )
                    print(pos_gap_indices, zero_gap_indices)

                    frac = np.divide((upper_y - lower_y)[pos_gap_indices], (upper_x - lower_x)[pos_gap_indices])
                    n_pos = pos_gap_indices.shape[0]
                    A = np.zeros((n_pos, n_pos))
                    # print(A)
                    for i in range(n_pos):
                        A[i, i] = frac[i, 0]
                    b = np.multiply(frac, -lower_x[pos_gap_indices]) + lower_y[pos_gap_indices]

                    # print(xxT_var.shape)
                    # print(xxT_var[pos_gap_indices].shape)
                    # print(xxT_var[pos_gap_indices][:, pos_gap_indices].shape)
                    # print(xxT_var[pos_gap_indices][:, pos_gap_indices])
                    # print(xxT_var[pos_gap_indices, pos_gap_indices].shape)
                    # exit(0)

                    constr = A @ x_var[pos_gap_indices] @ upper_x[pos_gap_indices].T - \
                        A @ xxT_var[pos_gap_indices][:, pos_gap_indices] + \
                        b @ upper_x[pos_gap_indices].T - \
                        b @ x_var[pos_gap_indices].T - \
                        y_var[pos_gap_indices] @ upper_x[pos_gap_indices].T + \
                        yxT_var[pos_gap_indices][:, pos_gap_indices]

                    # print(constr, constr.shape)
                    constraints += [constr >= 0]
                    # print(lower_x[zero_gap_indices])
                    # exit(0)

                    # A = np.diag(frac)
                    # print(frac)
                    # print(A, A.shape)

                    # for idx in zero_gap_indices:
                    #     # constraints += [x_var[idx] == upper_x[idx]]
                    #     constraints += [y_var[idx] == np.maximum(upper_x[idx], 0)]
                    #     constraints += [yyT_var[idx, idx] == (np.maximum(upper_x[idx], 0)) ** 2]

                    # TODO: only planet for pos_gap_indices, for the rest, we should just be able to leave them alone?
                    # exit(0)
                else:
                    # frac = np.divide((upper_y - lower_y)[pos_gap_indices], (upper_x - lower_x)[pos_gap_indices])
                    frac = np.divide(upper_y - lower_y, upper_x - lower_x)
                    # n = pos_gap_indices.shape[0]
                    A = np.zeros((n, n))
                    for i in range(n):
                        # this absolutely should not have been necessary, but I couldn't get the numpy functions
                        # to work as desired
                        A[i, i] = frac[i, 0]
                    b = np.multiply(frac, -lower_x) + lower_y

                    # exit(0)

                    # constraints += [
                    #     y_var <= A @ x_var + b,
                    #     yyT_var <= A @ xxT_var @ A.T + A @ x_var @ b.T + b @ x_var.T @ A.T + b @ b.T
                    # ]

                    constraints += [
                        cp.diag(A @ x_var @ upper_x.T - A @ xxT_var + b @ upper_x.T -
                                b @ x_var.T - y_var @ upper_x.T + yxT_var) >= 0,

                        # cp.diag()

                        # cp.diag(A @ xxT_var - A @ x_var @ lower_x.T + b @ x_var.T - b @ lower_x.T \
                        # - yxT_var + y_var @ lower_x.T) >= 0,
                        # cp.diag(
                        #     A @ xxT_var + A @ x_var @ b.T - A @ yxT_var.T + b @ x_var.T @ A.T + b @ b.T \
                        #  - b @ y_var.T - yxT_var @ A.T - y_var @ b.T + yyT_var
                        # ) >= 0,
                    ]
    return constraints


def nonneg_orthant_proj_bound_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    step = steps[i]
    #  prev_step = steps[i - 1]
    y = step.get_output_var()
    x = step.get_input_var()
    n = y.get_dim()
    lower_x = curr.iterate_vars[x].get_lower_bound()
    upper_x = curr.iterate_vars[x].get_upper_bound()
    # print(lower_x, upper_x)
    zeros = np.zeros((n, 1))
    lower_y = np.maximum(lower_x, zeros)
    upper_y = np.maximum(upper_x, zeros)
    # print(lower_y, upper_y)
    curr.iterate_vars[y].set_lower_bound(lower_y)
    curr.iterate_vars[y].set_upper_bound(upper_y)
