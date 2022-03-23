import numpy as np
import cvxpy as cp
import gurobipy as gp
import scipy.sparse as spa
from qcqp2quad_form.quad_extractor import QuadExtractor
from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from quadratic_functions.extended_slemma_sdp import *
from gurobipy import GRB


def eval_lasso(A, b, x, lambd):
    c = A @ x - b
    return .5 * c.T @ c + lambd * np.linalg.norm(x, 1)


def form_problem_and_extract():
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

    x0 = cp.Variable(n)
    x1 = cp.Variable(n)
    u = cp.Variable(n)
    gamma1 = cp.Variable(n)
    gamma2 = cp.Variable(n)

    obj = .5 * cp.quad_form(x1, A.T @ A) - b.T @ A @ x1 + lambd_ones.T @ u + .5 * b.T @ b
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

    constraints.append(u >= 0)

    # Eq 1: x1 + (t A^TA - I)x0 + gamma_1 - gamma_2 - tA^Tb = 0
    constraints.append(x1 - (I - t * A.T @ A) @ x0 + gamma1 - gamma2 == t * A.T @ b)

    # Eq 2: gamma_1 + gamma_2 - lambda*t*ones = 0
    constraints.append(gamma1 + gamma2 == lambdt_ones)

    # Eq 3: gamma_1.T x1 - gamma_1.T u = 0
    constraints.append(gamma1.T @ (x1 - u) == 0)

    # Eq 4: gamma_2.T x1 + gamma_2.T u = 0
    constraints.append(gamma2.T @ (x1 + u) == 0)

    # Fake constraints
    # constraints.append(cp.quad_form(x1, I) <= R ** 2)
    # constraints.append(cp.quad_form(u, I) <= R ** 2)

    problem = cp.Problem(cp.Minimize(-obj), constraints)
    quad_extractor = QuadExtractor(problem)
    return quad_extractor


def test_ISTA_SDR():
    quad_extractor = form_problem_and_extract()
    data_objective = quad_extractor.extract_objective()
    data_constraints, data_constr_types = quad_extractor.extract_constraints()
    # print(data_objective)
    # for i in range(len(data_constraints)):
    #     print('Constraint', i+1, data_constraints[i], data_constr_types[i])
    #
    # print(quad_extractor.inverse_data)
    # print(t * A.T @ b)
    # print(problem.variables())
    # print(quad_extractor.inverse_data.get_var_offsets(problem.variables()))
    # for c in problem.constraints:
    #     print(type(c))
    #
    # for c in quad_extractor.symb_problem.constraints:
    #     print(c, type(c))
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


def test_ISTA_Gurobi():
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
    m.setObjective(obj, GRB.MAXIMIZE)

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
    test_ISTA_SDR()
    # test_ISTA_Gurobi()


if __name__ == '__main__':
    main()
