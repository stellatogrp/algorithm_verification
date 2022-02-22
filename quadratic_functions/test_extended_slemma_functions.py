import numpy as np
import cvxpy as cp

from extended_slemma_quadpep import *


def l_inf_ball_constraints(n, R=1):
    '''
        Generate list of constraints for l_inf ball of radius R in dim n centered at 0 TODO: general center

        To bound each component, we need to bound each component $|x_i| \leq R$,
            or $x_i - R \leq 0$ and $ -x_i - R \leq 0$

        Each constraint will be generated as: $x^T H_i x + 2c_i^T x + d_i \leq 0$
    '''
    I = np.eye(n)
    constraints = []
    zeros = np.zeros((n, n))
    for i in range(n):
        # constraints.append((zeros, .5 * I[i], -R))
        # constraints.append((zeros, -.5 * I[i], -R))
        constraints.append((np.outer(I[i], I[i]), np.zeros(n), -R ** 2))
    return constraints


def form_l_inf_ball_sdp():
    n = 3
    mu = 2
    L = 20
    R = 1
    gamma = 2 / (mu + L)
    I = np.eye(n)
    P = np.array([[mu, 0, 0], [0, L, 0], [0, 0, (mu + L) / 2]])
    H0 = -(gamma ** 2) * P @ P + 2 * gamma * P - I
    c0 = np.zeros(n)
    d0 = 0
    obj_list = (H0, c0, d0)
    constraint_param_lists = l_inf_ball_constraints(n, R=1)
    # solve_extended_slemma_sdp_ineq_only(n, obj_list, constraint_param_lists)
    solve_full_extended_slemma_dual_sdp(n, obj_list, ineq_param_lists=constraint_param_lists)


def test_single_constraint_sdp():
    n = 3
    mu = 2
    L = 20
    R = 1
    gamma = 2 / (mu + L)
    I = np.eye(n)
    P = np.array([[mu, 0, 0], [0, L, 0], [0, 0, (mu + L) / 2]])
    print(P)
    Qb = -(gamma ** 2) * P @ P + 2 * gamma * P - I
    print(Qb)
    ub = np.zeros(n)
    cb = 0

    Qa = I
    ua = np.zeros(n)
    ca = -R ** 2
    obj_param_list = (Qb, ub, cb)
    cons = [(Qa, ua, ca)]
    solve_extended_slemma_sdp_ineq_only(n, obj_param_list, cons)


def main():
    # form_and_solve_basic_sdp()
    # test_single_constraint_sdp()
    form_l_inf_ball_sdp()


if __name__ == '__main__':
    main()
