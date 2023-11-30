import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from UQP_class import UnconstrainedQuadraticProgram

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 20,})


def solve_sdp(UQP, c, r, k=1):
    n = UQP.n
    P = UQP.P
    t = UQP.get_t_opt()
    x = cp.Variable(n)
    x_mat = cp.reshape(x, (n, 1))
    X = cp.Variable((n, n), symmetric=True)
    # C = np.
    ItP = np.eye(n) - t * P
    # print(np.linalg.matrix_power(ItP, 0))
    C = np.linalg.matrix_power(ItP, k) - np.linalg.matrix_power(ItP, k-1)
    CTC = C.T @ C
    # print(CTC)

    obj = cp.Maximize(cp.trace(CTC @ X))
    constraints = [
        cp.trace(X) <= r ** 2 + 2 * c.T @ x - c.T @ c,
        cp.bmat([
            [X, x_mat],
            [x_mat.T, np.array([[1]])]
        ]) >> 0,
    ]

    prob = cp.Problem(obj, constraints)
    res = prob.solve()
    # print(res)
    # print(X.value)
    U, sigma, VT = np.linalg.svd(X.value, full_matrices=False)
    # print('eigvals of X:', sigma)
    # print(U, U[:, 0])
    # print('Q:', UQP.Q)
    return res, U[:, 0] * np.sqrt(sigma[0])


def gd(x0, t, k, UQP):
    out = [x0]
    curr = x0
    for _ in range(k):
        new = curr - t * UQP.gradf(curr)
        out.append(new)
        # print(new)
        curr = new
    return out


def UQP_pep(mu, L, r, t, k=1):
    verbose=0
    problem = PEP()
    func = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    xs = func.stationary_point()
    x0 = problem.set_initial_point()

    x = [x0 for _ in range(k + 1)]
    for i in range(k):
        x[i+1] = x[i] - t * func.gradient(x[i])

    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)

    problem.set_performance_metric((x[-1] - x[-2]) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    return pepit_tau


def experiment3():
    seed1 = 5
    seed2 = 6
    n = 2
    mu = 1
    L = 10
    R = .9
    r = .1
    k = 1
    gd_k = 10

    # np.random.seed(seed)
    UQP1 = UnconstrainedQuadraticProgram(n, mu=mu, L=L, seed=seed1, centered=False)
    UQP2 = UnconstrainedQuadraticProgram(n, mu=mu, L=L, seed=seed2, centered=False)
    print(UQP1.P, UQP2.P)

    z0 = np.array([0.9, 0])
    res1, x1 = solve_sdp(UQP1, z0, r)
    x1 = -x1
    print(res1, x1)

    res2, x2 = solve_sdp(UQP2, z0, r)
    x2 = -x2
    print(res2, x2)

    def f1_plot(*args):
        x = np.array([x_i for x_i in args])
        return UQP1.f(x)

    def f2_plot(*args):
        x = np.array([x_i for x_i in args])
        return UQP2.f(x)

    x_min = -1.1
    x_max = 1.1
    y_min = -1.1
    y_max = 1.1

    x_lin = np.linspace([x_min, y_min], [x_max, y_max], 100)
    f1_vec = np.vectorize(f1_plot)
    f2_vec = np.vectorize(f2_plot)
    X1, X2 = np.meshgrid(x_lin[:, 0], x_lin[:, 1])
    x_star = np.array([0, 0])

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.contour(X1, X2, f1_vec(X1, X2), colors='k', linestyles='solid')
    plt.contour(X1, X2, f2_vec(X1, X2), colors='k', linestyles='dashed')

    ax.scatter(*zip(x_star), marker='*', s=600, color='k')

    circ = plt.Circle(z0, r, fill=False, color='black', linestyle='dotted')
    ax.add_patch(circ)

    gd1_out = gd(x1, UQP1.get_t_opt(), gd_k, UQP1)
    gd2_out = gd(x2, UQP2.get_t_opt(), gd_k, UQP2)

    fp1_resids = [np.linalg.norm(gd1_out[i+1] - gd1_out[i]) ** 2 for i in range(gd_k)]
    fp2_resids = [np.linalg.norm(gd2_out[i+1] - gd2_out[i]) ** 2 for i in range(gd_k)]

    ax.plot(*zip(*gd1_out), linestyle='solid', marker='<',
            markerfacecolor='none', label=r'$P_1$')

    ax.plot(*zip(*gd2_out), linestyle='dashed', marker='>',
            markerfacecolor='none', label=r'$P_2$')

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig('experiment3/exp3contours.pdf')

    taus = []
    pep_r = R + r
    for k in range(1, gd_k + 1):
        tau = UQP_pep(mu, L, pep_r, UQP1.get_t_opt(), k=k)
        taus.append(tau)
    print(taus)

    labels = [r'$P_1$', r'$P_2$', 'PEP']
    markers = ['<', '>', 'x']

    plt.cla()
    plt.clf()
    plt.close()
    fig, ax = plt.subplots()

    ax.set_yscale('log')
    ax.set_xlabel('$K$')
    ax.set_ylabel('Worst case fixed point residual')
    K_vals = range(1, gd_k+1)
    # ax.set_title('NNLS SDP Relaxation')
    for (resids, label, marker) in zip([fp1_resids, fp2_resids, taus], labels, markers):
        ax.plot(K_vals, resids, label=label, marker=marker, markerfacecolor='none')

    plt.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig('experiment3/exp3resids.pdf')


def main():
    experiment3()


if __name__ == '__main__':
    main()
