import numpy as np
import cvxpy as cp
import gurobipy as gp
import scipy.sparse as spa
from quadratic_functions.extended_slemma_sdp import *
from cvxpy.utilities.coeff_extractor import CoeffExtractor
from cvxpy.reductions import InverseData
from cvxpy.reductions.qp2quad_form.qp2symbolic_qp import accepts, Qp2SymbolicQp
from cvxpy.reductions.qp2quad_form.qp_matrix_stuffing import QpMatrixStuffing, ParamQuadProg
from gurobipy import GRB


def block_vector_to_components(A, b):
    '''
    Given a block set of equations/inequalities: Ax + b = 0
    Return a list of triples corresponding to the component wise equation or inequality
    I.e. Ax + b = 0 becomes [(Z, A_i^T, b_i)]
        Z for the zero quadratic term since the intention is to feed into SDP relaxation
        A_i^T is ith row of A, b_i is ith component of b

    '''
    n = b.shape[0]
    Z = spa.csc_matrix(np.zeros((3 * n, 3 * n)))
    triple_list = []
    for i in range(n):
        Ai = A[i]
        bi = b[i]
        triple_list.append((Z, Ai, bi))
    return triple_list


def NNLS_PEP_SDP_onestep():
    np.random.seed(0)

    n = 2
    m = 3
    lambd = 5
    t = .05
    R = 1

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    # I = spa.eye(n)
    I = np.eye(n)
    Z = np.zeros((n, n))

    print(A, b)
    n_SDP = 3 * n
    NZ = spa.csc_matrix(np.zeros((n_SDP, n_SDP)))

    # The stacked variable vector will be x = [x1, y1, x0]

    # obj
    Hobj = np.zeros((n_SDP, n_SDP))
    Hobj[0:n, 0:n] = .5 * A.T @ A
    Hobj = spa.csc_matrix(Hobj)

    cobj = np.zeros(n_SDP)
    cobj[0:n] = -A.T @ b
    # cobj[2 * n:3 * n] = np.zeros(n)
    # cobj = spa.csc_matrix(cobj)
    # print(cobj)

    dobj = .5 * b.T @ b.T

    obj_triple = (-Hobj, -cobj, -dobj)
    ineq_triples = []
    eq_triples = []

    # Ineq 1: -x^1 \leq 0
    ineq1_block = np.block([-I, Z, Z])
    ineq1_lists = block_vector_to_components(ineq1_block, np.zeros(n))
    ineq_triples += ineq1_lists

    # Ineq 2: y^1 - x^1 \leq 0
    ineq2_block = np.block([-I, I, Z])
    ineq2_lists = block_vector_to_components(ineq2_block, np.zeros(n))
    ineq_triples += ineq2_lists

    # Ineq 3: x0^T x0 - R^2 \leq 0
    ineq3_mat = np.zeros((n_SDP, n_SDP))
    ineq3_mat[2*n:3*n, 2*n:3*n] = np.eye(n)
    ineq_triples.append((spa.csc_matrix(ineq3_mat), np.zeros(n_SDP), -R ** 2))

    # Eq 1:  y1 + (t A^TA - I)x0 - tA^Tb = 0
    eq1_block = np.block([Z, I, t * A.T @ A - I])
    eq1_lists = block_vector_to_components(eq1_block, -t * A.T @ b)
    eq_triples += eq1_lists

    # Eq 2:
    for i in range(n):
        H = spa.csc_matrix(np.zeros((n_SDP, n_SDP)))
        H[i, i] = 1
        H[i, n+i] = -1
        H = (H + H.T) / 2
        print(H)
        eq_triples.append((H, np.zeros(n_SDP), 0))


    result, M = solve_full_extended_slemma_primal_sdp(n_SDP, obj_triple, ineq_param_lists=ineq_triples, eq_param_lists=eq_triples)
    # print(np.round(M, 4))

    m = gp.Model()
    m.setParam('NonConvex', 2)

    x = m.addMVar(n_SDP,
                  name='x',
                  ub=gp.GRB.INFINITY * np.ones(n_SDP),
                  lb=-gp.GRB.INFINITY * np.ones(n_SDP))

    obj = x @ Hobj @ x + cobj @ x + dobj
    m.setObjective(obj, GRB.MAXIMIZE)
    for (Hi, ci, di) in ineq_triples:
        m.addConstr(x @ Hi @ x + ci @ x + di <= 0)

    for (Hj, cj, dj) in eq_triples:
        m.addConstr(x @ Hj @ x + cj @ x + dj == 0)

    m.optimize()
    print(x.X)


def main():
    NNLS_PEP_SDP_onestep()


if __name__ == '__main__':
    main()
