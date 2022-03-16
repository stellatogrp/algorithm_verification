import numpy as np
import cvxpy as cp
import gurobipy as gp
import scipy.sparse as spa
from qcqp2quad_form.quad_extractor import QuadExtractor
from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from quadratic_functions.extended_slemma_sdp import solve_full_extended_slemma_primal_sdp, solve_full_extended_slemma_dual_sdp
from gurobipy import GRB


def basic_gurobi_test():
    n = 1
    m = gp.Model()
    I = np.eye(n)

    x = m.addMVar(n,
                  name='x',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))

    obj = x @ I @ x
    m.setObjective(obj, GRB.MAXIMIZE)
    m.addConstr(-x <= 0)
    m.addConstr(x - 1 <= 0)
    m.optimize()
    print('optimal x:', x.X)


def basic_KKT_test():
    print('--------testing basic KKT with Gurobi---------')
    n = 1
    I = np.eye(n)
    minusI = -I

    m = gp.Model()
    m.setParam('NonConvex', 2)

    x = m.addMVar(n,
                  name='x',
                  ub=gp.GRB.INFINITY * np.ones(n),
                  lb=-gp.GRB.INFINITY * np.ones(n))
    gamma1 = m.addMVar(n,
             name='gamma1',
             ub=gp.GRB.INFINITY * np.ones(n),
             lb=-gp.GRB.INFINITY * np.ones(n))
    gamma2 = m.addMVar(n,
                       name='gamma2',
                       ub=gp.GRB.INFINITY * np.ones(n),
                       lb=-gp.GRB.INFINITY * np.ones(n))

    m.setObjective(x @ minusI @ x, GRB.MAXIMIZE)
    m.addConstr(-2 * x - gamma1 + gamma2 == 0)
    m.addConstr(gamma1 >= 0)
    m.addConstr(gamma2 >= 0)
    # m.addConstr(-x <= 0)
    m.addConstr(-x + .5 <= 0)
    m.addConstr(x - 1 <= 0)
    m.addConstr(gamma1 @ (-x) == 0)
    m.addConstr(gamma2 @ (x - 1) == 0)

    m.optimize()
    print('x:', x.X, gamma1.X, gamma2.X)


def test_quad_extractor():
    print('--------testing quad extractor--------')
    n = 1
    I = np.eye(n)
    minusI = -I

    x = cp.Variable(n)
    gamma1 = cp.Variable(n)
    gamma2 = cp.Variable(n)

    obj = cp.quad_form(x, minusI)
    constraints = []

    constraints.append(-2 * x - gamma1 + gamma2 == 0)
    constraints.append(-gamma1 <= 0)
    constraints.append(-gamma2 <= 0)
    constraints.append(-x + .5 <= 0)
    constraints.append(x - 1 <= 0)
    constraints.append(gamma1 @ (-x) == 0)
    constraints.append(gamma2 @ (x - 1) == 0)

    problem = cp.Problem(cp.Maximize(obj), constraints)
    quad_extractor = QuadExtractor(problem)
    data_objective = quad_extractor.extract_objective()
    data_constraints, data_constr_types = quad_extractor.extract_constraints()

    k = data_objective['q'].shape[0]
    Hobj = data_objective['P']
    cobj = data_objective['q']
    dobj = data_objective['r']

    print('----testing quad extractor with Gurobi----')
    print(k)
    m = gp.Model()
    m.setParam('NonConvex', 2)

    y = m.addMVar(k,
                  name='y',
                  ub=gp.GRB.INFINITY * np.ones(k),
                  lb=-gp.GRB.INFINITY * np.ones(k))

    obj = y @ Hobj @ y + cobj @ y + dobj
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
                m.addConstr(ccons @ y + dcons <= 0)
            else:
                m.addConstr(y @ Hcons @ y + ccons @ y + dcons <= 0)
        else:
            # eq_triples.append((Hcons, ccons, dcons))
            if Hcons is None:
                m.addConstr(ccons @ y + dcons == 0)
            else:
                m.addConstr(y @ Hcons @ y + ccons @ y + dcons == 0)

    m.optimize()
    print(y.X)


def test_basic_quadratic_driver():
    n = 1
    print('--------initial opt with gurobi--------')
    basic_gurobi_test()
    basic_KKT_test()
    test_quad_extractor()


def main():
    test_basic_quadratic_driver()


if __name__ == '__main__':
    main()
