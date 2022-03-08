import numpy as np
import scipy.sparse as spa
from quadratic_functions.extended_slemma_sdp import *


def block_vector_to_components(A, b):
    '''
    Given a block set of equations/inequalities: Ax + b = 0
    Return a list of triples corresponding to the component wise equation or inequality
    I.e. Ax + b = 0 becomes [(Z, A_i^T, b_i)]
        Z for the zero quadratic term since the intention is to feed into SDP relaxation
        A_i^T is ith row of A, b_i is ith component of b

    '''
    n = b.shape[0]
    Z = spa.csc_matrix(np.zeros((5 * n, 5 * n)))
    triple_list = []
    for i in range(n):
        Ai = A[i]
        bi = b[i]
        triple_list.append((Z, Ai, bi))
    return triple_list


def ISTA_PEP_SDP_onestep():
    np.random.seed(0)

    n = 2
    m = 3
    lambd = 0
    t = .05
    R = 1

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    lambdt_ones = lambd * t * np.ones(n)
    # I = spa.eye(n)
    I = np.eye(n)
    Z = np.zeros((n, n))

    # print(A, b)
    n_SDP = 5 * n
    NZ = spa.csc_matrix(np.zeros((n_SDP, n_SDP)))

    # The stacked variable vector will be x = [x0, x1, u, gamma_1, gamma_2]

    # obj
    Hobj = np.zeros((n_SDP, n_SDP))
    Hobj[n:2 * n, n:2 * n] = .5 * A.T @ A
    Hobj = spa.csc_matrix(Hobj)

    cobj = np.zeros(n_SDP)
    cobj[n:2 * n] = -A.T @ b
    cobj[2 * n:3 * n] = lambdt_ones
    # cobj[2 * n:3 * n] = np.zeros(n)
    # cobj = spa.csc_matrix(cobj)
    print(cobj)

    dobj = .5 * b.T @ b.T

    obj_triple = (-Hobj, -cobj, -dobj)
    ineq_triples = []
    eq_triples = []

    # Ineq 1: -gamma_1 \leq 0
    ineq1_block = np.block([Z, Z, Z, -I, Z])
    ineq1_lists = block_vector_to_components(ineq1_block, np.zeros(n))
    ineq_triples += ineq1_lists

    # Ineq 2: -gamma_2 \leq 0
    ineq2_block = np.block([Z, Z, Z, Z, -I])
    ineq2_lists = block_vector_to_components(ineq2_block, np.zeros(n))
    ineq_triples += ineq2_lists

    # Ineq 3: x^1 - u \leq 0
    ineq3_block = np.block([Z, I, -I, Z, Z])
    ineq3_lists = block_vector_to_components(ineq3_block, np.zeros(n))
    ineq_triples += ineq3_lists

    # Ineq 4: -x^1 - u \leq 0
    ineq4_block = np.block([Z, -I, -I, Z, Z])
    ineq4_lists = block_vector_to_components(ineq4_block, np.zeros(n))
    ineq_triples += ineq4_lists

    # Ineq 5: x0^T x0 - R^2 \leq 0
    ineq5_mat = np.zeros((n_SDP, n_SDP))
    ineq5_mat[0:n, 0:n] = np.eye(n)
    ineq_triples.append((spa.csc_matrix(ineq5_mat), np.zeros(n_SDP), -R ** 2))

    # Eq 1: x1 + (t A^TA - I)x0 + gamma_1 - gamma_2 - tA^Tb = 0
    eq1_block = np.block([t*A.T @ A - I, I, Z, I, -I])
    eq1_lists = block_vector_to_components(eq1_block, -t * A.T @ b)
    eq_triples += eq1_lists

    # Eq 2: gamma_1 + gamma_2 - lambda*t*ones = 0
    eq2_block = np.block([Z, Z, Z, I, I])
    eq2_lists = block_vector_to_components(eq2_block, -lambdt_ones)
    eq_triples += eq2_lists

    # Eq 3: gamma_1.T x1 - gamma_1.T u = 0
    eq3_mat = np.zeros((n_SDP, n_SDP))
    eq3_mat[n:2 * n, 3 * n: 4 * n] = I
    eq3_mat[2 * n:3 * n, 3 * n: 4 * n] = -I
    eq3_mat = spa.csc_matrix(.5 * (eq3_mat + eq3_mat.T))
    eq_triples.append((eq3_mat, np.zeros(n_SDP), 0))

    # Eq 4: gamma_2.T x1 + gamma_2.T u = 0
    eq4_mat = np.zeros((n_SDP, n_SDP))
    eq4_mat[n:2 * n, 4 * n:5 * n] = I
    eq4_mat[2 * n:3 * n, 4 * n:5 * n] = I
    eq4_mat = spa.csc_matrix(.5 * (eq4_mat + eq4_mat.T))
    eq_triples.append((eq4_mat, np.zeros(n_SDP), 0))

    result, M = solve_full_extended_slemma_primal_sdp(n_SDP, obj_triple, ineq_param_lists=ineq_triples, eq_param_lists=eq_triples)
    print(np.round(M, 4))


def main():
    ISTA_PEP_SDP_onestep()


if __name__ == '__main__':
    main()
