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
    lambd = 5
    t = .05
    R = 1

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    lambd_ones = lambd * np.ones(n)
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
    cobj[2 * n:3 * n] = lambd_ones
    # cobj[2 * n:3 * n] = np.zeros(n)
    # cobj = spa.csc_matrix(cobj)
    # print(cobj)

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


def ISTA_PEP_SDP_onestep_alt_formulation():
    np.random.seed(0)

    n = 2
    m = 3
    lambd = 1
    t = .05
    R = 1

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    lambdt_ones = lambd * t * np.ones(n)
    # I = spa.eye(n)
    I = np.eye(n)
    Z = np.zeros((n, n))

    x0 = cp.Variable(n)
    x1 = cp.Variable(n)
    u = cp.Variable(n)
    gamma1 = cp.Variable(n)
    gamma2 = cp.Variable(n)

    obj = .5 * cp.quad_form(x1, A.T @ A) - b.T @ A @ x1 + lambdt_ones.T @ u + .5 * b.T @ b
    constraints = []

    # Ineq 5: x0^T x0 - R^2 \leq 0
    constraints.append(cp.quad_form(x0, I) <= R ** 2)

    # Ineq 1: gamma_1 \geq 0
    constraints.append(gamma1 >= 0)

    # Ineq 2: -gamma_2 \leq 0
    constraints.append(gamma2 >= 0)

    # Ineq 3: x^1 - u \leq 0
    constraints.append(x1 <= u)

    # Ineq 4: -x^1 - u \leq 0
    constraints.append(-x1 <= u)

    # Eq 1: x1 + (t A^TA - I)x0 + gamma_1 - gamma_2 - tA^Tb = 0
    constraints.append(x1 - (I - t*A.T @ A) @ x0 + gamma1 - gamma2 == t*A.T @ b)

    # Eq 2: gamma_1 + gamma_2 - lambda*t*ones = 0
    constraints.append(gamma1 + gamma2 == lambdt_ones)

    # Eq 3: gamma_1.T x1 - gamma_1.T u = 0
    constraints.append(gamma1.T @ x1 - gamma1.T @ u == 0)

    # Eq 4: gamma_2.T x1 + gamma_2.T u = 0
    constraints.append(gamma2.T @ x1 + gamma2.T @ u == 0)

    print(A.T @ A)
    problem = cp.Problem(cp.Maximize(obj), constraints)
    # problem.solve()
    # print(problem.variables())
    inverse_data = InverseData(problem)
    extractor = CoeffExtractor(inverse_data)
    expr = problem.objective.expr.copy()
    print(expr)
    print(extractor.x_length)
    # print(inverse_data.get_var_offsets(problem.variables()))

    print('--------qp2symbolicqp--------')
    symbolic = Qp2SymbolicQp(problem)
    data, solving_chain, inverse_data = problem.get_problem_data(cp.MOSEK)


def main():
    ISTA_PEP_SDP_onestep()
    # ISTA_PEP_SDP_onestep_alt_formulation()


if __name__ == '__main__':
    main()
