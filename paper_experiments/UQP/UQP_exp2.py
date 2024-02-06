import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from UQP_class import UnconstrainedQuadraticProgram

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.size": 16,
})


def solve_sdp(UQP, c, r, z0, k=1):
    n = UQP.n
    P = UQP.P
    t = UQP.get_t_opt()
    x = cp.Variable(n)
    x_mat = cp.reshape(x, (n, 1))
    X = cp.Variable((n, n), symmetric=True)
    # C = np.
    ItP = np.eye(n) - t * P
    # print(np.linalg.matrix_power(ItP, 0))
    A = np.linalg.matrix_power(ItP, k) - np.linalg.matrix_power(ItP, k-1)
    # ATA = A.T @ A
    # print(CTC)
    B = -t * np.linalg.matrix_power(ItP, k-1)
    BTB = B.T @ B
    Az = A @ z0

    obj = cp.Maximize(cp.trace(BTB @ X) + 2 * Az.T @ B @ x + Az.T @ Az)
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
    print('eigvals of X:', sigma)
    # print(U, U[:, 0])
    # print('Q:', UQP.Q)
    return res, U[:, 0] * np.sqrt(sigma[0])


def gd(x0, t, k, q, UQP):
    P = UQP.P
    out = [x0]
    curr = x0
    for _ in range(k):
        new = curr - t * (P @ curr + q)
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


def experiment2():
    seed = 0
    n = 2
    mu = 1
    L = 10
    # R = .9
    r = .25
    k = 1
    gd_k = 10
    z0 = np.array([.5, .5])
    UQP = UnconstrainedQuadraticProgram(n, mu=mu, L=L, seed=seed, centered=True)

    q1 = np.array([1, 0])
    q2 = np.array([0, 1])

    # q1 = np.array([np.sqrt(2)/2, -np.sqrt(2)/2])
    # q2 = np.array([np.sqrt(2)/2, np.sqrt(2)/2])

    res1, q1_wc = solve_sdp(UQP, q1, r, z0, k=k)
    if np.linalg.norm(q1 - q1_wc) > r + 1e-3:
        print('flipping q1')
        q1_wc = -q1_wc
    print('q1:', res1, q1_wc, np.linalg.norm(q1 - q1_wc))

    q1_opt = -np.linalg.solve(UQP.P, q1_wc)

    res2, q2_wc = solve_sdp(UQP, q2, r, z0, k=k)
    if np.linalg.norm(q2 - q2_wc) > r + 1e-3:
        print('flipping q2')
        q2_wc = -q2_wc
    print('q2:', res2, q2_wc)

    q2_opt = -np.linalg.solve(UQP.P, q2_wc)

    y_min = -.75
    y_max = .75
    x_min = -1.5
    x_max = .75

    print('z0 to q1 opt:', np.linalg.norm(z0 - q1_opt))
    print('z0 to q2 opt:', np.linalg.norm(z0 - q2_opt))

    x_lin = np.linspace([x_min, y_min], [x_max, y_max], 100)
    def f1_plot(*args):
        x = np.array([x_i for x_i in args])
        return UQP.f(x) + q1_wc @ x

    def f2_plot(*args):
        x = np.array([x_i for x_i in args])
        return UQP.f(x) + q2_wc @ x

    gd1_out = gd(z0, UQP.get_t_opt(), gd_k, q1_wc, UQP)
    gd2_out = gd(z0, UQP.get_t_opt(), gd_k, q2_wc, UQP)

    q1_resids = [np.linalg.norm(gd1_out[i+1] - gd1_out[i]) ** 2 for i in range(gd_k)]
    q2_resids = [np.linalg.norm(gd2_out[i+1] - gd2_out[i]) ** 2 for i in range(gd_k)]

    print(gd2_out, q2_opt)

    f1_vec = np.vectorize(f1_plot)
    f2_vec = np.vectorize(f2_plot)
    X1, X2 = np.meshgrid(x_lin[:, 0], x_lin[:, 1])

    fig, ax = plt.subplots(figsize=(6, 6))
    labels = [r'$z^\star_1$', r'$z^\star_2$']
    markers = ['<', '>']
    contour_levels = [0.25, 0.5]
    plt.contour(X1, X2, f1_vec(X1, X2), contour_levels, colors='k', linestyles='solid', alpha=0.25)
    plt.contour(X1, X2, f2_vec(X1, X2), contour_levels, colors='k', linestyles='solid', alpha=0.25)

    ax.scatter(*zip(q1_opt), marker='*', s=300, color='k', label=labels[0])
    ax.scatter(*zip(q2_opt), marker='s', s=300, color='k', label=labels[1])
    ax.scatter(*zip(z0), marker='x', s=300, color='k', label=r'$z^0$')

    # ax.plot(*zip(*gd_out), linestyle='--', marker=marker,
    #         markerfacecolor='none', label=label)
    ax.plot(*zip(*gd1_out), linestyle='--', marker=markers[0], markerfacecolor='None')
    ax.plot(*zip(*gd2_out), linestyle='--', marker=markers[1], markerfacecolor='None')

    pep_r = np.max([np.linalg.norm(z0 - q1_opt), np.linalg.norm(z0 - q2_opt)])

    circ = plt.Circle(q1_opt, pep_r, fill=True, alpha=0.1, color='black', linestyle='dotted')

    ax.add_patch(circ)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig('experiment2/exp2contours.pdf')

    # PEP
    taus = []
    for k in range(1, gd_k + 1):
        tau = UQP_pep(mu, L, pep_r, UQP.get_t_opt(), k=k)
        taus.append(tau)
    print(taus)
    print(q1_resids, q2_resids)

    labels = ['q1 worst case', 'q2 worst case', 'PEP']
    labels = [r'$q^\star_1$', r'$q^\star_2$', r'$\mathrm{PEP}$']
    markers = ['<', '>', 'o']
    colors = ['tab:blue', 'tab:orange', 'g']

    plt.cla()
    plt.clf()
    plt.close()
    fig, ax = plt.subplots()

    ax.set_yscale('log')
    ax.set_xlabel('$K$')
    ax.set_ylabel('Worst case fixed point residual')
    K_vals = range(1, gd_k+1)
    # ax.set_title('NNLS SDP Relaxation')
    for (resids, label, marker, color) in zip([q1_resids, q2_resids, taus], labels, markers, colors):
        ax.plot(K_vals, resids, label=label, marker=marker, color=color, markerfacecolor='None')

    plt.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig('experiment2/exp2resids.pdf')


def main():
    experiment2()


if __name__ == '__main__':
    main()
