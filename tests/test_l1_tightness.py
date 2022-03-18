import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp

from gurobipy import GRB
from qcqp2quad_form.quad_extractor import QuadExtractor
from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from quadratic_functions.extended_slemma_sdp import *


def test_l1_GD_gurobi():
    print('-------directly testing l1 with gurobi--------')
    n = 2
    mu = 2
    L = 20
    R = 1
    gamma = 2 / (mu + L)
    I = spa.eye(n)
    P = spa.csc_matrix(np.array([[mu, 0], [0, L]]))
    ones = np.ones(n)

    Hobj = -(gamma ** 2) * P @ P + 2 * gamma * P - I
    cobj = np.zeros(n)
    dobj = 0

    m = gp.Model()
    m.setParam('NonConvex', 2)

    x = m.addMVar(n, name='x',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))

    u = m.addMVar(n, name='u')  # u represents l1 norm, so u >= 0 implicitly
    m.setObjective(x @ Hobj @ x + cobj @ x + dobj)
    m.addConstr(ones @ u <= R)
    m.addConstr(x <= u)
    m.addConstr(-x <= u)
    m.optimize()

    print(np.round(x.X, 4))


def get_quad_extractor():
    n = 2
    mu = 2
    L = 20
    R = 1
    gamma = 2 / (mu + L)
    I = spa.eye(n)
    P = spa.csc_matrix(np.array([[mu, 0], [0, L]]))
    ones = np.ones(n)

    x = cp.Variable(n)
    u = cp.Variable(n)

    Hobj = -(gamma ** 2) * P @ P + 2 * gamma * P - I
    cobj = np.zeros(n)
    dobj = 0

    obj = cp.quad_form(x, Hobj) + cobj @ x + dobj

    constraints = []

    constraints.append(ones @ u <= R)
    constraints.append(x <= u)
    constraints.append(-x <= u)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    quad_extractor = QuadExtractor(problem)
    return quad_extractor


def test_SDR_from_extractor():
    quad_extractor = get_quad_extractor()
    print('-------testing SDR---------')
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

    res, M = solve_homoegeneous_form_primal_sdp(n, obj_triple, ineq_param_lists=ineq_triples, eq_param_lists=eq_triples)
    print(res)


def test_SDR_with_extra_quadratics():
    quad_extractor = get_quad_extractor()
    print('-------testing SDR with extra quadratics---------')
    data_objective = quad_extractor.extract_objective()
    data_constraints, data_constr_types = quad_extractor.extract_constraints()
    n = data_objective['q'].shape[0]
    Hobj = data_objective['P']
    cobj = data_objective['q']
    dobj = data_objective['r']
    obj_triple = (Hobj, cobj, dobj)
    ineq_triples = []
    eq_triples = []

    for i in range(len(data_constraints)):
        constr = data_constraints[i]
        Hcons = constr['P']
        ccons = constr['q']
        dcons = constr['r']

        # add the original constraint
        constr_type = data_constr_types[i]
        if constr_type in (Inequality, NonPos, NonNeg):
            ineq_triples.append((Hcons, ccons, dcons))
        else:
            eq_triples.append((Hcons, ccons, dcons))

        # add the quadratic implied constraint
        if Hcons is None:
            m = ccons.shape[0]
            # print(m)
            for i in range(m):
                for j in range(i, m):
                    # print(i, j)
                    ai = ccons[i]
                    aj = ccons[j]
                    if m > 1:
                        bi = dcons[i]
                        bj = dcons[j]
                    else:
                        bi = dcons
                        bj = dcons
                    H = -np.outer(ai, aj)
                    c = bi * aj + bj * ai
                    d = -bi * bj
                    if constr_type in (Inequality, NonPos, NonNeg):
                        ineq_triples.append((H, c.reshape(1, c.shape[0]), d))
                    else:
                        eq_triples.append((H, c.reshape(1, c.shape[0]), d))

    res, M = solve_homoegeneous_form_primal_sdp(n, obj_triple, ineq_param_lists=ineq_triples, eq_param_lists=eq_triples)
    print(res)


def test_extra_quadratics_gurobi():
    print('--------testing extra quadratics with gurobi--------')
    quad_extractor = get_quad_extractor()
    data_objective = quad_extractor.extract_objective()
    data_constraints, data_constr_types = quad_extractor.extract_constraints()
    n = data_objective['q'].shape[0]
    Hobj = data_objective['P']
    cobj = data_objective['q']
    dobj = data_objective['r']
    obj_triple = (Hobj, cobj, dobj)

    m = gp.Model()
    m.setParam('NonConvex', 2)

    x = m.addMVar(n, name='x',
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

        if Hcons is None:
            k = ccons.shape[0]
            # print(m)
            for i in range(k):
                for j in range(i, k):
                    # print(i, j)
                    ai = ccons[i]
                    aj = ccons[j]
                    if k > 1:
                        bi = dcons[i]
                        bj = dcons[j]
                    else:
                        bi = dcons
                        bj = dcons
                    H = -np.outer(ai, aj)
                    c = bi * aj + bj * ai
                    d = -bi * bj
                    if constr_type in (Inequality, NonPos, NonNeg):
                        m.addConstr(x @ H @ x + c @ x + d <= 0)
                    else:
                        m.addConstr(x @ H @ x + c @ x + d == 0)

    m.optimize()
    print(x.X)


def main():
    test_l1_GD_gurobi()
    # test_SDR_from_extractor()
    # test_SDR_with_extra_quadratics()
    test_extra_quadratics_gurobi()


if __name__ == '__main__':
    main()
