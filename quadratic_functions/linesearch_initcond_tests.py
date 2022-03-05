import numpy as np
import cvxpy as cp

from extended_slemma_sdp import solve_full_extended_slemma_dual_sdp, solve_full_extended_slemma_primal_sdp


def general_l2_ball_constraints(xc, R=1):
    n = len(xc)
    return np.eye(n), -2*xc, np.dot(xc, xc) - R ** 2


def test_line_search_l_inf_ball():
    n = 2
    mu = 2
    L = 20
    R = 1

    I = np.eye(n)
    Z = np.zeros((n, n))
    P = np.array([[mu, 0], [0, L]])

    Hobj = -.5 * np.block([[Z, Z], [Z, P]])
    cobj = np.zeros(2 * n)
    dobj = 0

    # Hineq = .5 * np.block([[P, Z], [Z, Z]])
    Hineq = np.block([[I, Z], [Z, Z]])
    cineq = np.zeros(2 * n)
    dineq = -R ** 2

    Heq1 = np.block([[Z, -.5 * P], [-.5 * P.T, P]])
    ceq1 = np.zeros(2 * n)
    deq1 = 0

    Heq2 = np.block([[Z, .5 * P @ P], [.5 * P @ P, Z]])
    ceq2 = np.zeros(2 * n)
    deq2 = 0

    res, N = solve_full_extended_slemma_primal_sdp(2 * n, (Hobj, cobj, dobj), [(Hineq, cineq, dineq)],
                                                   [(Heq1, ceq1, deq1), (Heq2, ceq2, deq2)])

    print(np.round(N, decimals=4))
    print(np.linalg.matrix_rank(N))

    # xc = np.array([np.sqrt(2)/4, np.sqrt(2)/4])
    xc = np.array([.1, 0])
    Hcons, ccons, dcons = general_l2_ball_constraints(xc, R=1)
    Hineq2 = np.block([[Hcons, Z], [Z, Z]])
    cineq2 = np.block([ccons, np.zeros(n)])
    dineq2 = dcons

    print('non center at zero')
    res2, N2 = solve_full_extended_slemma_primal_sdp(2 * n, (Hobj, cobj, dobj), [(Hineq2, cineq2, dineq2)],
                                                   [(Heq1, ceq1, deq1), (Heq2, ceq2, deq2)])
    print(np.round(N2, decimals=4))
    # print(np.linalg.matrix_rank(N2))

    X = N2[:-1, :-1]
    x = N2[-1, :-1]
    # print(np.linalg.matrix_rank(X))
    print(np.outer(x, x))

    res3, M = solve_full_extended_slemma_dual_sdp(2 * n, (Hobj, cobj, dobj), [(Hineq2, cineq2, dineq2)],
                                                   [(Heq1, ceq1, deq1), (Heq2, ceq2, deq2)])


def main():
    test_line_search_l_inf_ball()


if __name__ == '__main__':
    main()
