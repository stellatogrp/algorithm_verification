import cvxpy as cp
import numpy as np
import pandas as pd
from NNLS import NNLS

# from PEPit.examples.composite_convex_minimization.proximal_gradient import wc_proximal_gradient
from PEPit import PEP
from PEPit.functions import ConvexFunction, SmoothStronglyConvexFunction
from PEPit.primitive_steps import proximal_step
from silver_strongcvx_NNLS import compute_silver_steps


def generate_samples(N, b_c, b_r, seed=2):
    np.random.seed(seed)
    out_b = []
    dim = b_c.shape
    for _ in range(N):
        normal_sample = np.random.normal(size=dim)
        normal_sample = normal_sample / np.linalg.norm(normal_sample) * np.random.uniform(0, b_r)
        b_sample = normal_sample + b_c
        # print(np.linalg.norm(b_sample - b_c))
        out_b.append(b_sample)
    return out_b


def x_opt_vals(A, b_samples):
    m, n = A.shape
    x = cp.Variable(n)
    b = cp.Parameter(m)
    prob = cp.Problem(cp.Minimize(cp.sum_squares(A @ x - b)), [ x>= 0])
    out_x = []
    for samp in b_samples:
        b.value = samp.reshape(-1, )
        prob.solve()
        out_x.append(x.value)
    return out_x


def single_NNLS_conv_resids(K_max, t_list, A, b):
    conv_resids = []
    m, n = A.shape

    xk = np.zeros(n)
    for i in range(K_max):
        print(t_list)
        t = t_list[i]
        D = np.eye(n) - t * A.T @ A
        c = t * A.T @ b.reshape(-1,)
        ykplus1 = D @ xk + c
        xkplus1 = np.maximum(ykplus1, 0)

        conv_resids.append(np.linalg.norm(xkplus1 - xk) ** 2)
        xk = xkplus1
    return conv_resids


def all_conv_resids(K_max, t_vals, t_opt, silvers, A, b_samples):
    out_res = []
    for i, b_samp in enumerate(b_samples):
        for t in t_vals:
            conv_resids = single_NNLS_conv_resids(K_max, [t] * K_max, A, b_samp)
            print(conv_resids)
            for l in range(len(conv_resids)):
                out_dict = dict(sample_num=i+1, K=l+1, t=t, resid=conv_resids[l])
                out_res.append(pd.Series(out_dict))
    out_df = pd.DataFrame(out_res)
    print(out_df)
    out_df.to_csv('data/strongcvx/sample_tfixed_m15n8.csv', index=False)

    out_res = []
    for i, b_samp in enumerate(b_samples):
        topt_conv_resids = single_NNLS_conv_resids(K_max, [t_opt] * K_max, A, b_samp)
        silver_conv_resids = single_NNLS_conv_resids(K_max, silvers, A, b_samp)
        for l in range(len(topt_conv_resids)):
            # first t opt
            out_dict = dict(sample_num=i+1, K=l+1, sched='t_opt', resid=topt_conv_resids[l])
            out_res.append(pd.Series(out_dict))

            # then silver
            out_dict = dict(sample_num=i+1, K=l+1, sched='silver', resid=silver_conv_resids[l])
            out_res.append(pd.Series(out_dict))

    out_df = pd.DataFrame(out_res)
    out_df.to_csv('data/strongcvx/sample_silver_m15n8.csv', index=False)


def single_pep_sample(t_list, mu, L, r, K):
    verbose=1
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
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    # fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Compute n steps of the Douglas-Rachford splitting starting from x0
    x = [x0 for _ in range(K + 1)]
    # w = [x0 for _ in range(N + 1)]
    # x = x0
    for i in range(K):
        t = t_list[i]
        y = x[i] - t * func2.gradient(x[i])
        x[i+1], _, _ = proximal_step(y, func1, t)

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)
    # problem.set_initial_condition((x[1] - x[0]) ** 2 <= r ** 2)

    # Set the performance metric to the final distance to the optimum in function values
    # problem.set_performance_metric((func2(y) + fy) - fs)
    problem.set_performance_metric((x[-1] - x[-2]) ** 2)
    # problem.set_performance_metric((x[-1] - xs) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    return pepit_tau


def all_pep_runs_tfixed(t_vals, mu, L, r, K_max):
    # tau_vals = []
    out_res = []
    for K in range(1, K_max+1):
        for t in t_vals:
            tau = single_pep_sample([t] * K_max, mu, L, r, K)
            out_dict = dict(t=t, K=K, tau=tau)
            out_res.append(pd.Series(out_dict))
    out_df = pd.DataFrame(out_res)
    print(out_df)
    out_df.to_csv('data/strongcvx/pep_tfixed_m15n8.csv', index=False)


def all_pep_runs_topt_silver(t_opt, silvers, mu, L, r, K_max):
    out_res = []
    for K in range(1, K_max + 1):
        # first t_opt
        tau = single_pep_sample([t_opt] * K_max, mu, L, r, K)
        # tau = single_pep_sample(silvers, mu, L, r, K)
        out_dict = dict(sched='t_opt', K=K, tau=tau)
        out_res.append(pd.Series(out_dict))

        # then silvers
        tau = single_pep_sample(silvers, mu, L, r, K)
        out_dict = dict(sched='silver', K=K, tau=tau)
        out_res.append(pd.Series(out_dict))
    out_df = pd.DataFrame(out_res)
    print(out_df)
    out_df.to_csv('data/strongcvx/pep_silver_m15n8.csv', index=False)


def main():
    # outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    # outf = outf_prefix + f'paper_experiments/NNLS/data/{curr_time}.csv'

    m, n = 15, 8
    bc_mul = 10
    b_c = bc_mul * np.ones((m, 1))
    b_r = 0.5
    K_max = 8
    seed = 5

    instance = NNLS(m, n, b_c, b_r, seed=seed)
    print(instance.L, instance.mu, instance.kappa)
    N = 50

    mu = instance.mu
    L = instance.L
    1 / L
    t_opt = instance.get_t_opt()
    t_max = 2 / L
    tval_eps = t_max - t_opt
    t_vals = [t_opt - 2 * tval_eps, t_opt - 1 * tval_eps, t_opt + 0.75 * tval_eps]

    A = instance.A
    b_samples = generate_samples(N, b_c, b_r, seed=2)
    x_opt = x_opt_vals(A, b_samples)
    max_x = max(x_opt, key=lambda x: np.linalg.norm(x))
    print(np.linalg.norm(max_x))
    max_r = np.linalg.norm(max_x)
    all_pep_runs_tfixed(t_vals, mu, L, max_r, K_max)

    t_opt = instance.get_t_opt()

    silvers = compute_silver_steps(instance.kappa, 2 ** int(np.ceil(np.log2(K_max))))
    silvers /= L
    silvers = list(silvers)[:K_max]

    all_pep_runs_topt_silver(t_opt, silvers, mu, L, max_r, K_max)
    all_conv_resids(K_max, t_vals, t_opt, silvers, A, b_samples)


if __name__ == '__main__':
    main()
