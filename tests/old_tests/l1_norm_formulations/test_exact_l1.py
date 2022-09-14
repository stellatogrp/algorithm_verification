import cvxpy as cp
import gurobipy as gp
import numpy as np


def test_l1_exact(x):
    print('--------solving gurobi formulation l1 norm--------')
    d = x.shape[0]

    model = gp.Model()
    model.setParam('NonConvex', 2)
    # model.setParam('MIPGap', .01)
    model.setParam('OutputFlag', 0)

    xplus = model.addMVar(d,
                          ub=gp.GRB.INFINITY * np.ones(d),
                          lb=np.zeros(d))

    xminus = model.addMVar(d,
                           ub=gp.GRB.INFINITY * np.ones(d),
                           lb=np.zeros(d))

    z = model.addMVar(d,
                      vtype=gp.GRB.BINARY)

    c = model.addVar()
    onesT = np.ones((1, d))

    model.addConstr(x == xplus - xminus)
    model.addConstr(onesT @ xplus + onesT @ xminus == c)
    # model.addConstr(xplus <= c * z)
    # model.addConstr(xminus <= c * (1 - z))
    for i in range(d):
        model.addConstr(xplus[i] <= c * z[i])
        model.addConstr(xminus[i] <= c * (1 - z[i]))

    model.setObjective(c, gp.GRB.MAXIMIZE)
    model.optimize()
    print(model.objVal)


def test_l1_SDP_basic(w):
    d = w.shape[0]
    x = w.reshape(-1, 1)
    xplus = cp.Variable((d, 1))
    xminus = cp.Variable((d, 1))
    ones_d = np.ones((d, 1))
    z = cp.Variable((d, 1))
    c = cp.Variable((1, 1))

    zzT = cp.Variable((d, d))
    zcT = cp.Variable((d, 1))
    ccT = cp.Variable((1, 1))

    constraints = [
        x == xplus - xminus,
        xplus >= 0, xminus >= 0,
        xplus <= zcT,
        xminus <= ones_d @ c - zcT,
        cp.sum(xplus) + cp.sum(xminus) == c,
        z == cp.reshape(cp.diag(zzT), (d, 1)),
        0 <= z, z <= 1, 0 <= zzT, zzT <= 1,
        0 <= zcT,
        cp.bmat([
            [zzT, zcT, z],
            [zcT.T, ccT, c],
            [z.T, c.T, np.array([[1]])]
        ]) >> 0,
        c <= 10,
    ]

    obj = cp.Maximize(c)
    prob = cp.Problem(obj, constraints)
    res = prob.solve(verbose=True)
    print(res)
    # print(ccT.value)
    # print(zzT.value)


def test_l1_SDP_full(x):
    print('--------testing full SDP with xplus and xminus included--------')
    d = x.shape[0]
    x_var = cp.Variable(d)
    xxT_var = cp.Variable((d, d))
    constraints = [
        x_var == x,
        xxT_var == np.outer(x, x)
    ]
    test_l1_SDP_full_inner(d, x_var, xxT_var, constraints, np.linalg.norm(x, 1))


def test_l1_SDP_variable_x(d):
    print('-------testing full SDP with x variable--------')
    x_var = cp.Variable(d)
    xxT_var = cp.Variable((d, d))

    # cp.reshape(cp.diag(xxT_var), (n, 1)) <= cp.multiply((l + u), x_var) - l * u,
    l = np.zeros(d)
    u = np.ones(d)
    constraints = [
        cp.diag(xxT_var) <= cp.multiply(l + u, x_var) - l * u,
    ]
    test_l1_SDP_full_inner(d, x_var, xxT_var, constraints, 10)


def test_l1_SDP_full_inner(d, x, xxT, constraints, max_val):
    M = cp.Variable((3 * d + 2, 3 * d + 2), symmetric=True)
    ones = np.ones((d, 1))
    # print(np.round(xxT, 4))
    constraints += [M >= 0, M >> 0, M[-1, -1] == 1]
    xplus = M[0: d, -1]
    xminus = M[d: 2 * d, -1]
    z = M[2 * d: 3 * d, -1]
    xplus_xplusT = M[0: d, 0: d]
    xplus_xminusT = M[0: d, d: 2 * d]
    xminus_xplusT = M[d: 2 * d, 0: d]
    xminus_xminusT = M[d: 2 * d, d: 2 * d]
    constraints += [
        x == xplus - xminus,
        xxT == xplus_xplusT - xplus_xminusT - xminus_xplusT + xminus_xminusT,
    ]
    zzT = M[2 * d: 3 * d, 2 * d: 3 * d]
    zcT = M[2 * d: 3 * d, -2]
    ccT = M[-2, -2]
    c = M[-2, -1]
    constraints += [
        xplus <= zcT,
        xminus <= cp.reshape(ones, (d,)) * c - zcT,
        c == cp.sum(xplus) + cp.sum(xminus),
        ccT == ones.T @ (xplus_xplusT + xplus_xminusT + xminus_xplusT + xminus_xminusT) @ ones,
        z == cp.diag(zzT),
        z <= 1, zzT <= 1,
        # ccT <= 50,
        ccT <= max_val ** 2,
        # ccT == max_val ** 2 - 1,
    ]

    obj = cp.Maximize(c)
    # obj = cp.Maximize(cp.sum(xplus))
    prob = cp.Problem(obj, constraints)
    res = prob.solve(verbose=False)
    print('result', res)
    print('xplus:', np.round(xplus.value, 4))
    print('xminus:', np.round(xminus.value, 4))
    print('z:', np.round(z.value, 4))
    print(c.value)


def test_l1(d):
    np.random.seed(2)
    x = np.random.randn(d)
    print(x)
    print('l1 norm:', np.linalg.norm(x, 1))
    # test_l1_exact(x)
    # test_l1_SDP_basic(x)
    test_l1_SDP_full(x)
    # test_l1_SDP_variable_x(d)


def main():
    d = 5
    test_l1(d=d)


if __name__ == '__main__':
    main()
