import numpy as np
import cvxpy as cp
import scipy.sparse as spa


def solve_extended_slemma_sdp_ineq_only(n, obj_param_list, constraint_param_lists):
    '''
        Utility function to solve the dual sdp with no equality constraints.
    '''
    return solve_full_extended_slemma_dual_sdp(n, obj_param_list, ineq_param_lists=constraint_param_lists)


def solve_full_extended_slemma_dual_sdp(n, obj_param_list, ineq_param_lists=None, eq_param_lists=None):
    '''
    Consider the general QCQP:
        min x.T @ H0 @ x + c0.T @ x + d0
        s.t. x.T @ Hi @ x + ci.T @ x + di <= 0, i \in I
             x.T @ Hj @ x + cj.T @ x + dj == 0, j \in J

    This function relaxes the general QCQP to an SDP using ideas from the S-Lemma. The S-Lemma says that when
        |I| = 1, |J|=0, subject to a regularity assumption, the relaxation is tight, regardless of positive
        semidefiniteness of the two quadratic terms. In other cases, this SDP still provides a lower bound.
        Note that this can be thought of as the dual approach to the SDP in solve_full_extended_slemma_primal_sdp.

    :param n: Dimension of the vectors to search over
    :param obj_param_list: Tuple (H0, c0, d0)
    :param ineq_param_lists: List of tuples [(Hi, ci, di)] for each inequality constraint
    :param eq_param_lists: List of tuples [(Hj, cj, dj)] for each equality constraint
    :return: The numerical result of the SDP as well as the M matrix
    '''
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

    M12_block *= .5
    M22_block -= eta

    constraints.append(M[0:n, 0:n] == M11_block)
    constraints.append(M[0:n, n] == M12_block)
    constraints.append(M[n][n] == M22_block)

    problem = cp.Problem(cp.Maximize(eta), constraints)
    result = problem.solve(solver=cp.MOSEK, verbose=True)

    print('dual sdp result:', result)
    return result, M.value


def solve_full_extended_slemma_primal_sdp(n, obj_param_list, ineq_param_lists=None, eq_param_lists=None):
    '''
    Consider the SDP:
        min Tr(H0 @ X) + c0.T @ x + d0
        s.t. x.T @ Hi @ x + ci.T @ x + di <= 0, i \in I
             x.T @ Hj @ x + cj.T @ x + dj == 0, j \in J
             X >> x @ x.T

    This function forms and solves a direct rank relaxation of a general QCQP as the above SDP.
    It can be thought of as the primal problem to the dual problem from solve_full_extended_slemma_dual_sdp

    :param n: Dimension of the vectors to search over
    :param obj_param_list: Tuple (H0, c0, d0)
    :param ineq_param_lists: List of tuples [(Hi, ci, di)] for each inequality constraint if applicable
    :param eq_param_lists: List of tuples [(Hj, cj, dj)] for each equality constraint if applicable
    :return: The numerical result of the SDP as well as the N matrix corresponding to the Schur complement constraint
        of X >> x @ x.T
    '''
    X = cp.Variable((n, n))
    x = cp.Variable(n)
    N = cp.Variable((n + 1, n + 1), symmetric=True)
    constraints = [N >> 0, N[0:n, 0:n] == X, N[0:n, n] == x, N[n][n] == 1]

    (H0, c0, d0) = obj_param_list
    obj = cp.trace(H0 @ X) + c0 @ x + d0

    if ineq_param_lists is not None:
        for i in range(len(ineq_param_lists)):
            Hi, ci, di = ineq_param_lists[i]
            if Hi is not None:
                # print('not none')
                constraints.append(cp.trace(Hi @ X) + ci @ x + di <= 0)
            else:
                # print('none')
                constraints.append(ci @ x + di <= 0)

    if eq_param_lists is not None:
        for j in range(len(eq_param_lists)):
            Hj, cj, dj = eq_param_lists[j]
            if Hj is not None:
                # print('eq not none')
                constraints.append(cp.trace(Hj @ X) + cj @ x + dj == 0)
            else:
                # print('eq none')
                constraints.append(cj @ x + dj == 0)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    result = problem.solve(solver=cp.MOSEK, verbose=True)
    print('primal sdp result', result)

    return result, N.value


def homogeneous_form(H, c, d):
    '''
        Returns homogeous form matrix M of the function so that (x, 1)^T M (x, 1) = f(x)
    '''
    # n = c.shape[0]
    # cmat = c.reshape(n, 1)
    return spa.bmat([[H, c/2], [c.T/2, d]])


def solve_homoegeneous_form_primal_sdp(n, obj_param_list, ineq_param_lists=None, eq_param_lists=None):
    X = cp.Variable((n + 1, n + 1), symmetric=True)

    (H0, c0, d0) = obj_param_list
    W = homogeneous_form(H0, c0.reshape(n, 1), d0)
    obj = cp.sum(cp.multiply(W, X))

    constraints = [X >> 0, X[-1, -1] == 1]

    if ineq_param_lists is not None:
        for (Hi, ci, di) in ineq_param_lists:
            for i in range(ci.shape[0]):
                c_val = ci[i]
                if ci.shape[0] > 1:
                    d_val = di[i]
                else:
                    d_val = di
                W = homogeneous_form(Hi, c_val.reshape(n, 1), d_val)
                constraints.append(cp.sum(cp.multiply(W, X)) <= 0)
            # W = homogeneous_form(Hi, ci, di)
            # constraints.append(cp.sum(cp.multiply(W, X)) <= 0)

    if eq_param_lists is not None:
        for (Hj, cj, dj) in eq_param_lists:
            for i in range(cj.shape[0]):
                c_val = cj[i]
                if cj.shape[0] > 1:
                    d_val = dj[i]
                else:
                    d_val = dj
                W = homogeneous_form(Hj, c_val.reshape(n, 1), d_val)
                constraints.append(cp.sum(cp.multiply(W, X)) == 0)
            # W = homogeneous_form(Hj, cj, dj)
            # constraints.append(cp.sum(cp.multiply(W, X)) == 0)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    result = problem.solve(solver=cp.MOSEK, verbose=True)

    return result, X


def main():
    pass


if __name__ == '__main__':
    main()
