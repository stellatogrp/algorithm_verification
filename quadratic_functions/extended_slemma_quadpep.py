import numpy as np
import cvxpy as cp


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


def solve_extended_slemma_sdp_ineq_only(n, obj_param_list, constraint_param_lists):
    return solve_full_extended_slemma_dual_sdp(n, obj_param_list, ineq_param_lists=constraint_param_lists)


def solve_full_extended_slemma_dual_sdp(n, obj_param_list, ineq_param_lists=None, eq_param_lists=None):
    eta = cp.Variable()
    M = cp.Variable((n + 1, n + 1), symmetric=True)
    constraints = [M >> 0]

    (H0, c0, d0) = obj_param_list
    M11_block = H0
    M12_block = c0
    M22_block = d0

    if ineq_param_lists is not None:
        lambd_dim = len(ineq_param_lists)
        lambd = cp.Variable(lambd_dim)
        for i in range(lambd_dim):
            (Hi, ci, di) = ineq_param_lists[i]
            M11_block += lambd[i] * Hi
            M12_block += lambd[i] * ci
            M22_block += lambd[i] * di
        constraints.append(lambd >= 0)

    if eq_param_lists is not None:
        kappa_dim = len(eq_param_lists)
        kappa = cp.Variable(kappa_dim)
        for j in range(kappa_dim):
            (Hj, cj, dj) = eq_param_lists[j]
            M11_block -= kappa[j] * Hj
            M12_block -= kappa[j] * cj
            M22_block -= kappa[j] * dj

    M22_block -= eta

    constraints.append(M[0:n, 0:n] == M11_block)
    constraints.append(M[0:n, n] == M12_block)
    constraints.append(M[n][n] == M22_block)

    problem = cp.Problem(cp.Maximize(eta), constraints)
    result = problem.solve()

    print('sdp result:', result)


def solve_full_extended_slemma_primal_sdp(n, obj_param_list, ineq_param_lists=None, eq_param_lists=None):
    X = cp.Variable((n, n))
    x = cp.Variable(n)
    N = cp.Variable((n + 1, n + 1), symmetric=True)
    constraints = [N >> 0, N[0:n, 0:n] == X, N[0:n, n] == x, N[n][n] == 1]

    (H0, c0, d0) = obj_param_list
    obj = cp.trace(H0 @ X) + 2 * c0 @ x + d0

    if ineq_param_lists is not None:
        for i in range(len(ineq_param_lists)):
            Hi, ci, di = ineq_param_lists[i]
            constraints.append(cp.trace(Hi @ X) + 2 * ci @ x + di <= 0)

    if eq_param_lists is not None:
        for j in range(len(eq_param_lists)):
            Hj, cj, dj = eq_param_lists[j]
            constraints.append(cp.trace(Hj @ X) + 2 * cj @ x + dj == 0)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    result = problem.solve()
    print('primal sdp result', result)


def main():
    # form_and_solve_basic_sdp()
    # test_single_constraint_sdp()
    form_l_inf_ball_sdp()


if __name__ == '__main__':
    main()
