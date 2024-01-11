import cvxpy as cp
import numpy as np
import pandas as pd
from ISTA_class import ISTA

# from PEPit.examples.composite_convex_minimization.proximal_gradient import wc_proximal_gradient
from PEPit import PEP
from PEPit.functions import ConvexLipschitzFunction, SmoothStronglyConvexFunction
from PEPit.primitive_steps import proximal_step


def sample_l2ball(c, r, N):
    all_samples = []
    for _ in range(N):
        sample = np.random.normal(0, 1, c.shape[0])
        sample = np.random.uniform(0, r) * sample / np.linalg.norm(sample)
        # print(np.linalg.norm(sample))
        # print(sample.reshape(-1, 1) + c)
        all_samples.append(sample.reshape(-1, 1) + c)
    return all_samples


def b_to_xopt(instance, b_vals):
    out = []
    A = instance.A
    lambd = instance.lambd

    for b in b_vals:
        x = cp.Variable(A.shape[1])
        obj = cp.Minimize(.5 * cp.sum_squares(A @ x - b.reshape(-1, )) + lambd * cp.sum(cp.abs(x)))
        prob = cp.Problem(obj, [])
        prob.solve()
        out.append(x.value)

    return out


def S(v, t):
    return np.maximum(v-t, 0) - np.maximum(-v-t, 0)


def ISTA_solve(instance, zk, b, K, t=.01):
    A = instance.A
    lambd = instance.lambd
    n = A.shape[1]
    I = np.eye(n)
    lhs = I - t * A.T @ A

    # zk = np.zeros(n)
    out_zres = []

    b = b.reshape(-1,)
    for _ in range(K):
        znew = S(lhs @ zk + t * A.T @ b, lambd * t)
        out_zres.append(np.linalg.norm(zk - znew) ** 2)
        zk = znew
    # print(out_zres)
    return out_zres


def FISTA_solve(instance, zk, b, K, t=.01):
    A = instance.A
    lambd = instance.lambd
    n = A.shape[1]
    I = np.eye(n)
    lhs = I - t * A.T @ A

    # zk = np.zeros(n)
    wk = zk.copy()

    out_zres = []
    b = b.reshape(-1,)

    beta_k = 1

    # K = 1000
    for _ in range(K):
        znew = S(lhs @ wk + t * A.T @ b, lambd * t)
        beta_new = .5 * (1 + np.sqrt(1 + 4 * beta_k ** 2))
        wnew = znew + (beta_k - 1) / beta_new * (znew - zk)

        out_zres.append(np.linalg.norm(zk - znew) ** 2)

        zk = znew
        beta_k = beta_new
        wk = wnew
    # print(np.round(zk, 4))
    # print(out_zres)
    return out_zres


def b_to_ISTA(instance, b_vals, ztest, K=7, t=.01):

    out_series = []
    for i, b in enumerate(b_vals):
        # print('ista:', ISTA_solve(instance, ztest, b, K))
        # print('fista:', FISTA_solve(instance, ztest, b, K))
        ISTA_res = ISTA_solve(instance, ztest, b, K)
        FISTA_res = FISTA_solve(instance, ztest, b, K)

        for j in range(K):
            out = {
                'sample': i+1,
                'K': j+1,
                'ista': ISTA_res[j],
                'fista': FISTA_res[j],
            }
            out_series.append(pd.Series(out))
    out_df = pd.DataFrame(out_series)
    out_df.to_csv('data/samples.csv', index=False)


def ista_pep(instance, r, K=7, t=.01):
    A = instance.A
    L = np.max(np.abs(np.linalg.eigvals(A.T @ A)))
    mu = np.min(np.abs(np.linalg.eigvals(A.T @ A)))
    print(L, mu)
    lambd = instance.lambd

    problem = PEP()
    f = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    h = problem.declare_function(ConvexLipschitzFunction, M=lambd)
    F = f + h

    zs = F.stationary_point()

    z0 = problem.set_initial_point()

    problem.set_initial_condition((z0 - zs) ** 2 <= r ** 2)

    z = [z0 for _ in range(K+1)]
    for i in range(K):
        z[i + 1], _, _ = proximal_step(z[i] - t * f.gradient(z[i]), h, t * lambd)

    problem.set_performance_metric((z[-1] - z[-2]) ** 2)
    pepit_tau = problem.solve(verbose=1)
    print(pepit_tau)
    return pepit_tau


def fista_pep(instance, r, K=7, t=.01):
    A = instance.A
    L = np.max(np.abs(np.linalg.eigvals(A.T @ A)))
    mu = np.min(np.abs(np.linalg.eigvals(A.T @ A)))
    print(L, mu)
    lambd = instance.lambd

    problem = PEP()
    f = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    h = problem.declare_function(ConvexLipschitzFunction, M=lambd)
    F = f + h

    zs = F.stationary_point()

    z0 = problem.set_initial_point()

    problem.set_initial_condition((z0 - zs) ** 2 <= r ** 2)

    beta = 1
    z = [z0 for _ in range(K + 1)]
    w = z0
    for i in range(K):
        z[i + 1], _, _ = proximal_step(w - t * f.gradient(w), h, t * lambd)
        beta_new = .5 * (1 + np.sqrt(1 + 4 * beta ** 2))
        w = z[i + 1] + (beta - 1) / beta_new * (z[i + 1] - z[i])
        beta = beta_new

    problem.set_performance_metric((z[-1] - z[-2]) ** 2)
    pepit_tau = problem.solve(verbose=1)
    print(pepit_tau)

    return pepit_tau


def r_to_pep(instance, r_max, K=7, t=.01):
    out_series = []

    for i in range(1, K+1):
        ista_tau = ista_pep(instance, r_max, K=i, t=.01)
        fista_tau = fista_pep(instance, r_max, K=i, t=.01)
        out = {
            't': t,
            'K': i,
            'ista_tau': ista_tau,
            'fista_tau': fista_tau,
        }
        out_series.append(pd.Series(out))
    out_df = pd.DataFrame(out_series)
    print(out_df)

    out_df.to_csv('data/pep.csv', index=False)


def sample_and_run(instance, b_c, b_r, N, ztest, t=.01, K=7):
    b_vals = sample_l2ball(b_c, b_r, N)
    x_opt_vals = b_to_xopt(instance, b_vals)

    print('z0 val:', np.round(ztest, 4))
    # max_r = np.linalg.norm(max(x_opt_vals, key=lambda x: np.linalg.norm(x - ztest)))
    max_x = max(x_opt_vals, key=lambda x: np.linalg.norm(x - ztest))

    print(np.round(max_x, 4))
    r_max = np.linalg.norm(max_x - ztest)
    print(r_max)

    b_to_ISTA(instance, b_vals, ztest, K=K, t=t)
    r_to_pep(instance, r_max, K=K)


def main():
    m, n = 20, 15
    b_c = 10 * np.ones((m, 1))
    b_r = .5
    lambd = 5
    N = 100
    K = 15

    instance = ISTA(m, n, b_c, b_r, lambd=lambd, seed=1)
    # ztest = instance.test_cp_prob()
    # print(ztest)
    ztest = np.zeros(n)
    # print(instance.get_t_opt())

    np.random.seed(0)
    sample_and_run(instance, b_c, b_r, N, ztest, K=K)


if __name__ == '__main__':
    main()
