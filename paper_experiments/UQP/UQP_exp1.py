import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from UQP_class import UnconstrainedQuadraticProgram

plt.rcParams.update({
    "text.usetex": True,
    "font.size": 16,})


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


def solve_centered_sdp(UQP, r, k=1):
    n = UQP.P.shape[0]
    return solve_sdp(UQP, np.zeros(n), r, k=k)


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


def sample_c(n, R):
    samp = np.random.randn(n)
    samp = R * samp / np.linalg.norm(samp)
    return samp


def off_centered_sdp(n, R, r, k, UQP):
    c = sample_c(n, R)
    res, x = solve_sdp(UQP, c, r, k=k)
    # print(res)
    # print(c, x)
    if np.linalg.norm(c - x) >= r + 1e-3:
        # print('flipping')
        x = -x
    # print(np.linalg.norm(c), np.linalg.norm(c - x))
    return c, x


def gd(x0, t, k, UQP):
    out = [x0]
    curr = x0
    for _ in range(k):
        new = curr - t * UQP.gradf(curr)
        out.append(new)
        # print(new)
        curr = new
    return out


def experiment1():
    seed = 1
    n = 2
    mu = 1
    L = 10
    R = .9
    r = .1
    k = 1
    gd_k = 10

    # np.random.seed(seed)
    UQP = UnconstrainedQuadraticProgram(n, mu=mu, L=L, seed=seed, centered=True)
    res, x_wc = solve_centered_sdp(UQP, R + r, k=k)
    print(x_wc)
    c_wc = R * x_wc / np.linalg.norm(x_wc)
    # print('centered at 0 res:', res)
    c1, x1 = off_centered_sdp(n, R, r, k, UQP)
    c2, x2 = off_centered_sdp(n, R, r, k, UQP)
    # UQP_pep(mu, L, R+r, UQP.get_t_opt(), k=k)

    x_min = -1.1
    x_max = 1.1
    y_min = -1.1
    y_max = 1.1

    x_lin = np.linspace([x_min, y_min], [x_max, y_max], 100)
    def f_plot(*args):
        x = np.array([x_i for x_i in args])
        return UQP.f(x)

    f_vec = np.vectorize(f_plot)
    X1, X2 = np.meshgrid(x_lin[:, 0], x_lin[:, 1])
    x_star = np.array([0, 0])

    fig, ax = plt.subplots(figsize=(6, 6))
    contour_levels = [0.25, 0.5, 1, 2, 3, 4, 5]
    plt.contour(X1, X2, f_vec(X1, X2), contour_levels, colors='k', linestyles='solid', alpha=0.25)

    circ1 = plt.Circle((0, 0), 1, fill=False, color='black')

    ax.add_patch(circ1)

    ax.scatter(*zip(x_star), marker='*', s=600, color='k')

    labels = [r'$\mathrm{worst~case}$', r'$c_1$', r'$c_2$']
    markers = ['^', '<', '>']
    fp_resids = []
    for (x, c, label, marker) in zip([x_wc, x1, x2], [c_wc, c1, c2], labels, markers):
        print(x, c)
        circ = plt.Circle(c, r, fill=False, color='black', linestyle='dotted')

        gd_out = gd(x, UQP.get_t_opt(), gd_k, UQP)
        fp_resids.append([np.linalg.norm(gd_out[i+1] - gd_out[i]) ** 2 for i in range(gd_k)])
        print(fp_resids)

        ax.add_patch(circ)
        ax.plot(*zip(*gd_out), linestyle='--', marker=marker,
            markerfacecolor='none', label=label)

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig('experiment1/exp1contours.pdf')

    # PEP
    taus = []
    for k in range(1, gd_k + 1):
        tau = UQP_pep(mu, L, R + r, UQP.get_t_opt(), k=k)
        taus.append(tau)

    fp_resids.append(taus)
    labels.append(r'$\mathrm{PEP}$')
    markers.append('x')

    plt.cla()
    plt.clf()
    plt.close()
    fig, ax = plt.subplots()
    # ax.plot(K_vals, silver_objs, label='silver', marker='>')

    # ax.plot(K_vals, topt_objs, label=f'$t^\star = {t_opt:.3f}$', marker='^')

    ax.set_yscale('log')
    ax.set_xlabel('$K$')
    ax.set_ylabel('Worst case fixed point residual')
    K_vals = range(1, gd_k+1)
    # ax.set_title('NNLS SDP Relaxation')
    for (resids, label, marker) in zip(fp_resids, labels, markers):
        ax.plot(K_vals, resids, label=label, marker=marker, markerfacecolor='none')

    plt.tight_layout()
    plt.legend()
    # plt.show()
    plt.savefig('experiment1/exp1resids.pdf')


def main():
    experiment1()


if __name__ == '__main__':
    main()
