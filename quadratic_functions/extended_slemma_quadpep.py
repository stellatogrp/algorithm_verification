import numpy as np
import cvxpy as cp


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
    pass


if __name__ == '__main__':
    main()
