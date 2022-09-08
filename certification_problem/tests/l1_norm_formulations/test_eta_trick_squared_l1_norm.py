import numpy as np
import gurobipy as gp
import cvxpy as cp


def eta_trick(w):
    print('--------testing direct eta trick formulation--------')
    d = w.shape[0]

    model = gp.Model()
    model.setParam('NonConvex', 2)
    # model.setParam('MIPGap', .01)
    model.setParam('OutputFlag', 0)

    eta = model.addMVar(d,
                        ub=gp.GRB.INFINITY * np.ones(d),
                        lb=np.zeros(d))

    eta_inv = model.addMVar(d,
                            ub=gp.GRB.INFINITY * np.ones(d),
                            lb=np.zeros(d))

    obj = 0
    for j in range(d):
        obj += (w[j] ** 2) * eta_inv[j]
        model.addConstr(eta[j] * eta_inv[j] == 1)
    model.addConstr(eta.sum() == 1)
    model.setObjective(obj)

    model.optimize()
    print('model obj:', model.objVal)
    print('eta:', eta.X, 'eta_inv:', eta_inv.X)


def eta_trick_basic_SDP(w):
    print('--------testing eta trick basic SDP--------')
    d = w.shape[0]
    w_sq = np.square(w)
    x_dim = 2 * d + 1
    ones = np.ones((d, 1))
    x = cp.Variable((x_dim, 1))
    X = cp.Variable((x_dim, x_dim), symmetric=True)
    M = cp.bmat([
        [X, x],
        [x.T, np.array([[1]])]
    ])
    constraints = [M >= 0, M >> 0]
    eta = M[-1, 0: d]
    alpha = M[-1, d: 2 * d]
    eta_alphaT = M[0: d, d: 2 * d]
    eta_etaT = M[0: d, 0: d]

    constraints += [
        cp.diag(eta_alphaT) == 1,
        ones.T @ eta == 1,
        ones.T @ eta_etaT @ ones == 1,
    ]

    obj = cp.Minimize(w_sq @ alpha)
    prob = cp.Problem(obj, constraints)
    res = prob.solve(verbose=True)
    print(res)


def eta_trick_KKT(w):
    print('--------testing eta trick KKT formulation--------')
    d = w.shape[0]

    model = gp.Model()
    model.setParam('NonConvex', 2)
    # model.setParam('MIPGap', .01)
    model.setParam('OutputFlag', 0)

    eta = model.addMVar(d,
                        ub=gp.GRB.INFINITY * np.ones(d),
                        lb=np.zeros(d))

    sigma = model.addMVar(d,
                          ub=gp.GRB.INFINITY * np.ones(d),
                          lb=np.zeros(d))

    lambd = model.addMVar(d,
                          ub=gp.GRB.INFINITY * np.ones(d),
                          lb=np.zeros(d))

    gamma = model.addVar(ub=gp.GRB.INFINITY,
                         lb=-gp.GRB.INFINITY)

    for j in range(d):
        model.addConstr(w[j] ** 2 + lambd[j] * sigma[j] == gamma * sigma[j])
        model.addConstr(sigma[j] == eta[j] ** 2)
        model.addConstr(lambd[j] * eta[j] == 0)

    model.addConstr(eta.sum() == 1)
    obj = 0
    model.setObjective(obj)

    model.optimize()
    print(model.objVal)
    print('eta:', eta.X)
    print('obj from eta vals:', np.sum(np.divide(w ** 2, eta.X)))


def eta_trick_KKT_norm_obj(w):
    print('--------testing eta trick KKT formulation with obj included--------')
    d = w.shape[0]

    model = gp.Model()
    model.setParam('NonConvex', 2)
    # model.setParam('MIPGap', .01)
    model.setParam('OutputFlag', 0)

    eta = model.addMVar(d,
                        ub=gp.GRB.INFINITY * np.ones(d),
                        lb=np.zeros(d))

    alpha = model.addMVar(d,
                          ub=gp.GRB.INFINITY * np.ones(d),
                          lb=np.zeros(d))

    sigma = model.addMVar(d,
                          ub=gp.GRB.INFINITY * np.ones(d),
                          lb=np.zeros(d))

    lambd = model.addMVar(d,
                          ub=gp.GRB.INFINITY * np.ones(d),
                          lb=np.zeros(d))

    gamma = model.addVar(ub=gp.GRB.INFINITY,
                         lb=-gp.GRB.INFINITY)

    for j in range(d):
        model.addConstr(w[j] ** 2 + lambd[j] * sigma[j] == gamma * sigma[j])
        model.addConstr(sigma[j] == eta[j] ** 2)
        model.addConstr(lambd[j] * eta[j] == 0)
        model.addConstr(alpha[j] * eta[j] == 1)

    model.addConstr(eta.sum() == 1)
    obj = 0
    for j in range(d):
        obj += (w[j] ** 2) * alpha[j]
    model.setObjective(obj)

    model.optimize()
    print('obj:', model.objVal)
    print('eta:', eta.X)
    # print('obj from eta vals:', np.sum(np.divide(w ** 2, eta.X)))


def eta_trick_KKT_SDP(w):
    d = w.shape[0]
    w_sq = np.square(w)
    # print(w, w_sq)
    x_dim = 4 * d + 1
    x = cp.Variable((x_dim, 1))
    eta = x[0: d]
    alpha = x[d: 2 * d]
    sigma = x[2 * d: 3 * d]
    lambd = x[3 * d: 4 * d]
    gamma = x[4 * d]

    X = cp.Variable((x_dim, x_dim), symmetric=True)


def test_eta_trick_squared_l1(d=2):
    np.random.seed(2)
    x = np.random.randn(d)
    print(x)
    print('squared l1 norm:', np.linalg.norm(x, 1) ** 2)
    # eta_trick(x)
    eta_trick_basic_SDP(x)
    # eta_trick_KKT(x)
    # eta_trick_KKT_norm_obj(x)
    # eta_trick_KKT_SDP(x)


def main():
    d = 5
    test_eta_trick_squared_l1(d=d)


if __name__ == '__main__':
    main()
