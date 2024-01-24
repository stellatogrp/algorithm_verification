from datetime import datetime

import cvxpy as cp
import numpy as np
import pandas as pd
from NNLS_class import NNLS

# from PEPit.examples.composite_convex_minimization.proximal_gradient import wc_proximal_gradient
from PEPit import PEP
from PEPit.functions import (
    ConvexFunction,
    SmoothStronglyConvexFunction,
)
from tqdm import tqdm


def generate_all_t_vals(t_vals, num_between=2):
    t_min, t_opt, t_max = t_vals
    t_min_to_opt = np.logspace(np.log10(t_min), np.log10(t_opt), num=num_between+1)
    t_opt_to_max = np.logspace(np.log10(t_opt), np.log10(t_max), num=num_between+1)
    # print(t_min_to_opt)
    # print(t_opt_to_max)
    t_out = np.hstack([t_min_to_opt, t_opt_to_max[1:]])
    # print(t_out)
    return t_out


def generate_samples(N, b_c, b_r, seed=2):
    np.random.seed(seed)
    out_b = []
    dim = b_c.shape
    for _ in range(N):
        normal_sample = np.random.normal(size=dim)
        normal_sample = normal_sample / np.linalg.norm(normal_sample) * np.random.uniform(0, b_r)
        b_sample = normal_sample + b_c
        print(np.linalg.norm(b_sample - b_c))
        out_b.append(b_sample)
    return out_b


def x_opt_vals(A, b_samples):
    m, n = A.shape
    x = cp.Variable(n)
    b = cp.Parameter(m)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [x >= 0])
    out_x = []
    for samp in b_samples:
        b.value = samp.reshape(-1, )
        prob.solve()
        out_x.append(x.value)
    return out_x


def single_NNLS_conv_resids(K_max, t, A, b):
    conv_resids = []
    m, n = A.shape
    D = np.eye(n) - t * A.T @ A
    c = t * A.T @ b.reshape(-1,)
    xk = np.zeros(n)
    for _ in range(K_max):
        ykplus1 = D @ xk + c
        xkplus1 = np.maximum(ykplus1, 0)

        conv_resids.append(np.linalg.norm(xkplus1 - xk) ** 2)
        xk = xkplus1
    return conv_resids


def all_conv_resids(K_max, t_vals, A, b_samples):
    out_res = []
    for i, b_samp in tqdm(enumerate(b_samples)):
        for t in t_vals:
            conv_resids = single_NNLS_conv_resids(K_max, t, A, b_samp)
            # print(conv_resids)
            for l in range(len(conv_resids)):
                out_dict = dict(sample_num=i+1, K=l+1, t=t, resid=conv_resids[l])
                out_res.append(pd.Series(out_dict))
    out_df = pd.DataFrame(out_res)
    print(out_df)
    # print('NOT OVERWRITING SAMPLES DATA')
    # out_df.to_csv('data/sample_data.csv', index=False)


def single_pep_sample(t, mu, L, r, K, test_opt_dist = False):
    verbose=2
    problem = PEP()
    print(mu, L)
    # L = 74.659
    # mu = .1
    # t = 2 / (L + mu)
    # r = 1
    # K = 2

    # Declare a convex and a smooth convex function.
    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    # func2 = problem.declare_function(SmoothStronglyConvexQuadraticFunction, L=L, mu=mu)
    # Define the function to optimize as the sum of func1 and func2
    func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    # xs = func.stationary_point()
    xs = func2.stationary_point()
    # fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Compute n steps of the Douglas-Rachford splitting starting from x0
    x = [x0 for _ in range(K + 1)]
    # w = [x0 for _ in range(N + 1)]
    # x = x0
    for i in range(K):
        # y = x[i] - t * func2.gradient(x[i])
        # x[i+1], _, _ = proximal_step(y, func1, t)

        x[i + 1] = x[i] - t * func2.gradient(x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)
    # problem.set_initial_condition((x[1] - x[0]) ** 2 <= r ** 2)

    # Set the performance metric to the final distance to the optimum in function values
    # problem.set_performance_metric((func2(y) + fy) - fs)
    if test_opt_dist:
        problem.set_performance_metric((x[-1] - xs) ** 2)
    else:
        problem.set_performance_metric((x[-1] - x[-2]) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    mosek_params = {
        # 'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-5,
    }
    pepit_tau = problem.solve(verbose=pepit_verbose, solver=cp.MOSEK, mosek_params=mosek_params)

    return pepit_tau


def all_pep_runs(t_vals, mu, L, r, K_max):
    # tau_vals = []
    out_res = []
    for K in range(1, K_max+1):
        for t in t_vals:
            tau = single_pep_sample(t, mu, L, r, K)
            out_dict = dict(t=t, K=K, tau=tau)
            out_res.append(pd.Series(out_dict))
            out_df = pd.DataFrame(out_res)
            print(out_df)
            # out_df.to_csv('data/pep_data.csv', index=False)


def main():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
    outf_prefix = '/home/vranjan/algorithm-certification/'
    # outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/NNLS/data/{curr_time}.csv'
    print(outf)

    m, n = 60, 40
    b_cmul = 30
    b_c = b_cmul * np.ones((m, 1))
    b_c[30:] = 0
    b_r = 0.5
    # K = 5
    # K_vals = [1, 2, 3, 4, 6]
    # K_vals = [1]
    N = 10000

    instance = NNLS(m, n, b_c, b_r, ATA_mu=20, seed=1)
    print(instance.mu, instance.L, instance.kappa)
    print(instance.A)

    # exit(0)

    A = instance.A

    ATA = A.T @ A
    eigs = np.linalg.eigvals(ATA)
    mu = np.min(eigs)
    L = np.max(eigs)
    # all_t = generate_all_t_vals(instance.get_t_vals(), num_between=2)
    # print(all_t)

    all_t = np.array(instance.grid_t_vals())
    print('t_values:', all_t)

    # all_t = all_t[:-1]
    print(all_t)

    b_samples = generate_samples(N, b_c, b_r, seed=2)
    # print(b_samples)
    x_opt = x_opt_vals(A, b_samples)
    max_x = max(x_opt, key=lambda x: np.linalg.norm(x))
    print(np.linalg.norm(max_x))
    max_r = np.linalg.norm(max_x)

    # single_NNLS_conv_resids(10, all_t[0], A, b_samples[0])
    # all_conv_resids(10, all_t, A, b_samples)
    all_pep_runs(all_t, mu, L, max_r, 10)


if __name__ == '__main__':
    main()
