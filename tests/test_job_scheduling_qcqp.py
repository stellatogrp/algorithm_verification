import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import gurobipy as gp

from gurobipy import GRB
from qcqp2quad_form.quad_extractor import QuadExtractor
from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from quadratic_functions.extended_slemma_sdp import *


def form_scheduling_problem():
    '''
    max c.T @ z
    subject to Az \leq b
               0 \leq z\ leq d
               z \in \mathbb{Z}^n
    For simplicity, we use $d_i = 1$ for all $i$, so this simplifies to $z \in {0,1}$.
    As a QCQP constraint, we have $z \in {0,1}$ iff $z(1-z) = 0$.

    Generates a random z so that the problem will be feasible. Returns A, b, c.
    '''
    np.random.seed(0)

    n = 10
    m = 2
    max = 5

    A = np.random.uniform(0, max, (m, n))
    z = np.random.randint(2, size=n)
    b = A @ z
    c = np.random.uniform(0, max, n)

    return A, b, c


def test_job_scheduling_gurobi():
    A, b, c = form_scheduling_problem()
    n = c.shape[0]

    m = gp.Model()
    m.setParam('NonConvex', 2)
    x = m.addMVar(n, name='x',
                  ub=1 * np.ones(n))  # Can do vtype=GRB.BINARY

    m.setObjective(-c @ x)
    m.addConstr(A @ x <= b)
    m.addConstr(x @ (1 - x) == 0)  # each x_i is nonnegative, so this implies slackness

    m.optimize()
    print(x.X)


def solve_nonintegral_job_scheduling():
    A, b, c = form_scheduling_problem()
    n = c.shape[0]

    z = cp.Variable(n)
    obj = c.T @ z
    constraints = [A @ z <= b, z >= 0, z <= 1]

    problem = cp.Problem(cp.Minimize(-obj), constraints)
    result = problem.solve()
    return result


def get_quad_extractor():
    A, b, c = form_scheduling_problem()
    n = c.shape[0]

    z = cp.Variable(n)
    y = cp.Variable(n)
    obj = c.T @ z
    constraints = [A @ z <= b, z >= 0, z <= 1, z @ (1 - z) == 0]
    # constraints = [A @ z <= b, z @ (1 - z) == 0]
    # NOTE: if we remove the 0 \leq z \leq 1 constraints, MOSEK fails to solve the SDR
    # the primal and dual certificates terminate at around -2e4, which implies that the problem may actually be
    #       unbounded, but MOSEK cannot find this

    problem = cp.Problem(cp.Minimize(-obj), constraints)
    quad_extractor = QuadExtractor(problem)
    return quad_extractor


def test_job_scheduling_SDR():
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

    res, M = solve_homoegeneous_form_primal_sdp(n, obj_triple,
                                                ineq_param_lists=ineq_triples, eq_param_lists=eq_triples, verbose=False)
    print('SDP result:', res)
    relaxed_LP_res = solve_nonintegral_job_scheduling()
    print('relaxed LP result:', relaxed_LP_res)


def test_job_scheduling_with_diff_extractor():
    A, b, c = form_scheduling_problem()
    n = c.shape[0]
    binary_var_mat = np.array([[0, .5], [.5, 0]])

    z = cp.Variable(n)
    y = cp.Variable(n)
    t = cp.Variable((n, 2))
    obj = -c.T @ z
    constraints = [A @ z <= b, y == (1 - z), z >= 0, z <= 1]
    for i in range(n):
        constraints.append(t[i] == cp.reshape(cp.vstack([z[i], y[i]]), (2,)))
        constraints.append(t[i] @ binary_var_mat @ t[i] == 0)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    quad_extractor = QuadExtractor(problem)
    other_test(quad_extractor)


def other_test(quad_extractor):
    print('-------testing other SDR---------')
    data_objective = quad_extractor.extract_objective()
    data_constraints, data_constr_types = quad_extractor.extract_constraints()
    n = data_objective['q'].shape[0]
    print(n)

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

    res, M = solve_homoegeneous_form_primal_sdp(n, obj_triple,
                                                ineq_param_lists=ineq_triples, eq_param_lists=eq_triples, verbose=False)
    print('SDP result:', res)


def main():
    test_job_scheduling_gurobi()
    test_job_scheduling_SDR()
    # test_job_scheduling_with_diff_extractor()


if __name__ == '__main__':
    main()
