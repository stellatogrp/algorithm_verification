from datetime import datetime

import numpy as np
import pandas as pd
from NNLS import NNLS

# from PEPit.examples.composite_convex_minimization.proximal_gradient import wc_proximal_gradient
from PEPit import PEP
from PEPit.functions import (
    ConvexFunction,
    SmoothStronglyConvexQuadraticFunction,
)
from PEPit.primitive_steps import proximal_step
from silver_strongcvx_NNLS_sample import generate_samples, single_NNLS_conv_resids, x_opt_vals


def all_conv_resids(K_max, t, silvers, A, b_samples):
    out_res = []
    for i, b_samp in enumerate(b_samples):
        tconv_resids = single_NNLS_conv_resids(K_max, [t] * K_max, A, b_samp)
        silver_conv_resids = single_NNLS_conv_resids(K_max, silvers, A, b_samp)
        for l in range(len(tconv_resids)):
            out_dict = dict(sample_num=i+1, K=l+1, sched='silver', resid=silver_conv_resids[l])
            out_res.append(pd.Series(out_dict))
            out_dict = dict(sample_num=i+1, K=l+1, sched='fixed', resid=tconv_resids[l])
            out_res.append(pd.Series(out_dict))

    out_df = pd.DataFrame(out_res)
    print(out_df)
    out_df.to_csv('data/nonstrong_sample_data.csv', index=False)

    out_max = out_df.groupby(['sched', 'K']).max()
    # print(df.groupby(['t', 'K']).max())
    out_max.reset_index().to_csv('data/nonstrong_sample_max.csv', index=False)


def single_pep(t_list, L, r, K):
    verbose = 1
    problem = PEP()
    print(L)
    func1 = problem.declare_function(ConvexFunction)
    # func2 = problem.declare_function(SmoothConvexFunction, L=L)
    func2 = problem.declare_function(SmoothStronglyConvexQuadraticFunction, mu=0, L=L)

    func = func1 + func2

    xs = func.stationary_point()
    x0 = problem.set_initial_point()

    x = [x0 for _ in range(K + 1)]
    for i in range(K):
        t = t_list[i]
        y = x[i] - t * func2.gradient(x[i])
        x[i+1], _, _ = proximal_step(y, func1, t)

    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)
    problem.set_performance_metric((x[-1] - x[-2]) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    # pepit_tau = problem.solve(verbose=pepit_verbose)
    try:
        pepit_tau = problem.solve(verbose=pepit_verbose, wrapper='mosek')
    except AssertionError:
        pepit_tau = problem.objective.eval()

    return pepit_tau


def all_pep_runs(t, silvers, L, r, K_max):
    out_res = []
    for K in range(1, K_max + 1):
        tau = single_pep(silvers, L, r, K)
        out_dict = dict(sched='silver', K=K, tau=tau)
        out_res.append(pd.Series(out_dict))

        tau = single_pep([t] * K_max, L, r, K)
        out_dict = dict(sched='fixed', K=K, tau=tau)
        out_res.append(pd.Series(out_dict))
    out_df = pd.DataFrame(out_res)
    print(out_df)
    out_df.to_csv('data/nonstrong_pep_data_quad.csv', index=False)


def main():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
    outf_prefix = '/home/vranjan/algorithm-certification/'
    # outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/silver/data/{curr_time}.csv'
    print(outf)

    m, n = 60, 40
    # m, n = 10, 5
    b_cmul = 30
    b_c = b_cmul * np.ones((m, 1))
    b_c[30:] = 0
    # b_r = .5
    b_r = 0.5
    seed = 1
    N = 10000
    K_max = 7

    instance = NNLS(m, n, b_c, b_r, ATA_mu=0, seed=seed)
    A = instance.A
    print(instance.A)

    t = 1.5 / instance.L

    b_samples = generate_samples(N, b_c, b_r, seed=2)
    x_opt = x_opt_vals(A, b_samples)
    max_x = max(x_opt, key=lambda x: np.linalg.norm(x))
    print(np.linalg.norm(max_x))
    max_r = np.linalg.norm(max_x)

    print(max_r)
    silvers = instance.get_silver_steps(K_max)
    print(silvers)

    all_conv_resids(K_max, t, silvers, A, b_samples)
    all_pep_runs(t, silvers, instance.L, max_r, K_max)


if __name__ == '__main__':
    main()
