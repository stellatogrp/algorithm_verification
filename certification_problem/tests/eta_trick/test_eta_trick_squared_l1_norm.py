import numpy as np
import gurobipy as gp


def eta_trick(w):
    print('--------testing direct eta trick formulation--------')
    d = np.shape(w)[0]

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


def eta_trick_KKT(w):
    print('--------testing eta trick KKT formulation--------')
    d = np.shape(w)[0]

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


def test_eta_trick_squared_l1(d=2):
    np.random.seed(2)
    x = np.random.randn(d)
    print(x)
    print('squared l1 norm:', np.linalg.norm(x, 1) ** 2)
    eta_trick(x)
    eta_trick_KKT(x)


def main():
    d = 5
    test_eta_trick_squared_l1(d=d)


if __name__ == '__main__':
    main()
