import numpy as np
import cvxpy as cp
import gurobipy as gp
import scipy.sparse as spa
from qcqp2quad_form.quad_extractor import QuadExtractor
from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from quadratic_functions.extended_slemma_sdp import *
from gurobipy import GRB


def form_problem_and_extract():
    np.random.seed(0)

    n = 2
    m = 3
    t = .05
    R = 1

    A = np.random.randn(m, n)
    b = np.random.randn(m)
    # I = spa.eye(n)
    I = np.eye(n)

    print(A, b)

    x0 = cp.Variable(n)
    x1 = cp.Variable(n)
    gamma = cp.Variable(n)

    obj = .5 * cp.quad_form(x1, A.T @ A) - b.T @ A @ x1 + .5 * b.T @ b
    constraints = []

    constraints.append(cp.quad_form(x0, I) <= R ** 2)
    constraints.append(-gamma <= 0)
    constraints.append(-x1 <= 0)

    constraints.append(x1 - (I - t * A.T @ A) @ x0 - gamma == t * A.T @ b)
    constraints.append(gamma.T @ x1 == 0)

    problem = cp.Problem(cp.Minimize(-obj), constraints)
    quad_extractor = QuadExtractor(problem)
    return quad_extractor


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

    obj = x @ Hobj @ x + cobj @ x + dobj
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


def main():
    # test_NNLS_Gurobi()
    test_NNLS_SDR()


if __name__ == '__main__':
    main()
