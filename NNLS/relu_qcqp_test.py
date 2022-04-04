import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp
from qcqp2quad_form.quad_extractor import QuadExtractor
from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from quadratic_functions.extended_slemma_sdp import *

from gurobipy import GRB


def form_problem_and_extract():
    n = 5
    I = np.eye(n)
    np.random.seed(1)
    # y = np.random.randn(n)
    # print('original y:', np.round(y, 4))

    x = cp.Variable(n)
    y = cp.Variable(n)
    obj = y @ y
    constraints = []
    constraints.append(x >= 0)
    constraints.append(x >= y)
    constraints.append(cp.quad_form(y, I) <= 1)
    constraints.append(x.T @ (x - y) == 0)

    problem = cp.Problem(cp.Minimize(-obj), constraints)
    quad_extractor = QuadExtractor(problem)
    return quad_extractor


def test_NNLS_Gurobi():
    quad_extractor = form_problem_and_extract()
    data_objective = quad_extractor.extract_objective()
    data_constraints, data_constr_types = quad_extractor.extract_constraints()

    n = data_objective['q'].shape[0]
    Hobj = data_objective['P']
    cobj = data_objective['q']
    dobj = data_objective['r']

    print('-------- solving with gurobi --------')
    print(n)
    m = gp.Model()
    m.setParam('NonConvex', 2)

    x = m.addMVar(n,
                  name='x',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))

    if Hobj is not None:
        obj = x @ Hobj @ x + cobj @ x + dobj
    else:
        obj = cobj @ x + dobj
    m.setObjective(obj)

    for i in range(len(data_constraints)):
        constr = data_constraints[i]
        Hcons = constr['P']
        # if Hcons is None:
        #     Hcons = spa.csc_matrix(np.zeros((n, n)))
        ccons = constr['q']
        dcons = constr['r']
        constr_type = data_constr_types[i]
        if constr_type in (Inequality, NonPos, NonNeg):
            if Hcons is None:
                m.addConstr(ccons @ x + dcons <= 0)
            else:
                m.addConstr(x @ Hcons @ x + ccons @ x + dcons <= 0)
        else:
            # eq_triples.append((Hcons, ccons, dcons))
            if Hcons is None:
                m.addConstr(ccons @ x + dcons == 0)
            else:
                m.addConstr(x @ Hcons @ x + ccons @ x + dcons == 0)

    m.optimize()
    print(x.X)


def test_NNLS_SDR():
    quad_extractor = form_problem_and_extract()
    data_objective = quad_extractor.extract_objective()
    data_constraints, data_constr_types = quad_extractor.extract_constraints()

    n = data_objective['q'].shape[0]
    Hobj = data_objective['P']
    cobj = data_objective['q']
    dobj = data_objective['r']
    obj_triple = (Hobj, cobj, dobj)

    ineq_triples = []
    eq_triples = []

    # print(Hobj, cobj, dobj)
    for i in range(len(data_constraints)):
        constr = data_constraints[i]
        Hcons = constr['P']
        # if Hcons is None:
        #     Hcons = spa.csc_matrix(np.zeros((n, n)))
        ccons = constr['q']
        dcons = constr['r']
        constr_type = data_constr_types[i]
        if constr_type in (Inequality, NonPos, NonNeg):
            ineq_triples.append((Hcons, ccons, dcons))
        else:
            eq_triples.append((Hcons, ccons, dcons))

    # res, N = solve_full_extended_slemma_primal_sdp(n, obj_triple, ineq_param_lists=ineq_triples, eq_param_lists=eq_triples)
    # print(np.round(N, 4))
    res, M = solve_homoegeneous_form_primal_sdp(n, obj_triple, ineq_param_lists=ineq_triples, eq_param_lists=eq_triples)
    print(res)
    print(M.value)


def test_relu_Gurobi():
    n = 5
    I = np.eye(n)
    np.random.seed(1)
    y = np.random.randn(n)

    m = gp.Model()
    m.setParam('NonConvex', 2)

    x = m.addMVar(n, name='x')
    # y = m.addMVar(n,
    #               name='y',
    #               ub=gp.GRB.INFINITY * np.ones(n),
    #               lb=-gp.GRB.INFINITY * np.ones(n))
    # T = m.addMVar((n, n),
    #               name='T',
    #               ub=gp.GRB.INFINITY * np.ones((n, n)),
    #               lb=-gp.GRB.INFINITY * np.ones((n, n)))

    obj = 0
    m.setObjective(obj)
    m.addConstr(x >= y)
    m.addConstr(x >= 0)


    m.optimize()
    print('original y:', np.round(y, 4))
    print('x:', np.round(x.X, 4))


def test_specific_relu_SDP_n1():
    P = cp.Variable((3, 3), symmetric=True)
    np.random.seed(0)
    c = np.random.randn(1)
    l = -2
    u = -1

    obj = c * P[0, 2]
    constraints = []
    constraints.append(P[0, 2] >= 0)
    constraints.append(P[0, 2] >= P[1, 2])
    constraints.append(P[0, 0] == P[0, 1])
    constraints.append(P[1, 1] <= (l + u) * P[1, 2] - l * u)
    constraints.append(P[2, 2] == 1)
    constraints.append(P >> 0)

    problem = cp.Problem(cp.Maximize(obj), constraints)
    result = problem.solve(solver=cp.MOSEK, verbose=True)
    print('c:', c)
    print(result)
    print(P.value)
    
    
def test_specific_relu_SDP_n2():
    print('--------testing n=2--------')
    n = 2
    P = cp.Variable((5, 5), symmetric=True)
    np.random.seed(1)
    c = np.random.randn(2)
    l = -2
    u = -1
    
    obj = c.T @ P[0:2, -1]
    constraints = []
    constraints.append(P[0:2, -1] >= 0)
    constraints.append(P[0:2, -1] >= P[2:4, -1])

    constraints.append(P[0, 0] == P[0, 2])
    constraints.append(P[1, 1] == P[1, 3])

    constraints.append(P[2, 2] <= (l + u) * P[2, 4] - l * u)
    constraints.append(P[3, 3] <= (l + u) * P[3, 4] - l * u)

    constraints.append(P[-1, -1] == 1)
    constraints.append(P >> 0)

    problem = cp.Problem(cp.Maximize(obj), constraints)
    result = problem.solve(solver=cp.MOSEK, verbose=True)
    print('c:', c)
    print(result)
    print(np.round(P.value, 4))
    print(np.round(np.diag(P.value), 4))


def test_specific_relu_SDP_ndim():
    n = 2
    print('--------testing n=%d with general--------' % n)
    P = cp.Variable((2 * n + 1, 2 * n + 1), symmetric=True)
    np.random.seed(1)
    c = np.random.randn(n)
    l = -2
    u = -1

    obj = c.T @ P[0:n, -1]
    # obj = cp.sum_squares(P[0:n, -1])
    constraints = []
    constraints.append(P[0:n, -1] >= 0)
    constraints.append(P[0:n, -1] >= P[n:2*n, -1])

    constraints.append(cp.diag(P[0: n, 0: n]) == cp.diag(P[0: n, n: 2 * n]))
    constraints.append(cp.diag(P[n: 2 * n, n: 2 * n]) <= (l + u) * P[n: 2 * n, -1] - l * u)
    # constraints.append(cp.sum_squares(cp.diag(P[n: 2 * n, n: 2 * n])) <= 1)

    constraints.append(P[-1, -1] == 1)
    constraints.append(P >> 0)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    result = problem.solve(solver=cp.MOSEK, verbose=True)
    print('c:', c)
    print(result)
    print(np.round(P.value, 4))


def main():
    # test_relu_Gurobi()
    # test_NNLS_Gurobi()
    # test_NNLS_SDR()
    test_specific_relu_SDP_n1()
    test_specific_relu_SDP_n2()
    test_specific_relu_SDP_ndim()


if __name__ == '__main__':
    main()
