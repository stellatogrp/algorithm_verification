from datetime import datetime

import numpy as np
import pandas as pd
from NNLS import NNLS


def compute_shifted_2adics(K):
    return np.array([(k & -k).bit_length() for k in range(1, K+1)])


def compute_silver_idx(kappa, K):
    two_adics = compute_shifted_2adics(K)
    # print(two_adics)
    idx_vals = np.power(2, two_adics)

    # if np.ceil(np.log2(K)) == np.floor(np.log2(K)):
    last_pow2 = int(np.floor(np.log2(K)))
    # print(last_pow2)
    idx_vals[(2 ** last_pow2) - 1] /= 2
    print('a_idx:', idx_vals)
    return idx_vals


def generate_yz_sequences(kappa, t):
    # intended to be t = log_2(K)
    y_vals = {1: 1 / kappa}
    z_vals = {1: 1 / kappa}
    print(y_vals, z_vals)

    # first generate z sequences
    for i in range(1, t+1):
        K = 2 ** i
        z_ihalf = z_vals[int(K / 2)]
        xi = 1 - z_ihalf
        z_i = z_ihalf * (xi + np.sqrt(1 + xi ** 2))
        z_vals[K] = z_i

    for i in range(1, t+1):
        K = 2 ** i
        # z_ihalf = z_vals[int(K / 2)]
        # xi = 1 - z_ihalf
        # yi = z_ihalf / (xi + np.sqrt(1 + xi ** 2))
        # y_vals[K] = yi
        zK = z_vals[K]
        zKhalf = z_vals[int(K // 2)]
        yK = zK - 2 * (zKhalf - zKhalf ** 2)
        y_vals[K] = yK

    # print(y_vals, z_vals)

    # print(z_vals[1], z_vals[2])
    # print((1 / kappa ** 2) / z_vals[2])
    return y_vals, z_vals


def compute_silver_steps(kappa, K):
    # assume K is a power of 2
    idx_vals = compute_silver_idx(kappa, K)
    y_vals, z_vals = generate_yz_sequences(kappa, int(np.log2(K)))

    def psi(t):
        return (1 + kappa * t) / (1 + t)

    # print(y_vals, z_vals)
    silver_steps = []
    for i in range(idx_vals.shape[0] - 1):
        idx = idx_vals[i]
        silver_steps.append(psi(y_vals[idx]))
    silver_steps.append(psi(z_vals[idx_vals[-1]]))
    print(silver_steps)

    return np.array(silver_steps)


def silver_vs_opt_sdp(m, n, b_cmul, b_r, instance, K_max, seed):
    d = datetime.now()
    curr_time = d.strftime('%m%d%y_%H%M%S')
    outf_prefix = '/home/vranjan/algorithm-certification/'
    # outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/silver/data/{curr_time}.csv'
    print(outf)

    # K_vals = range(1, K_max + 1)
    K_vals = range(K_max, K_max + 1)

    # silvers = instance.get_silver_steps(K_max)
    kappa = instance.kappa
    silvers = compute_silver_steps(kappa, 2 ** int(np.ceil(np.log2(K_max))))
    silvers /= instance.L
    silvers = list(silvers)[:K_max]
    print(silvers)
    exit(0)
    t_opt = instance.get_t_opt()
    # exit(0)

    out_res = []
    for i, K in enumerate(K_vals):
        CP = instance.generate_CP(silvers, K)
        out = CP.solve(solver_type='SDP_CUSTOM')
        out['orig_m'] = m
        out['orig_n'] = n
        out['mu'] = instance.mu
        out['L'] = instance.L
        out['b_cmul'] = b_cmul
        out['b_r'] = b_r
        out['K'] = K
        out['sched'] = 'silver'
        out['t'] = silvers[i]
        out['seed'] = seed
        sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
        print(sdp_c, sdp_canontime, sdp_solvetime)

        # out_df = []
        out_res.append(pd.Series(out))
        out_df = pd.DataFrame(out_res)
        print(out_df)

        out_df.to_csv(outf, index=False)
        continue  # skip the fixed sched t

        CP2 = instance.generate_CP(t_opt, K)
        out = CP2.solve(solver_type='SDP_CUSTOM')
        out['orig_m'] = m
        out['orig_n'] = n
        out['mu'] = instance.mu
        out['L'] = instance.L
        out['b_cmul'] = b_cmul
        out['b_r'] = b_r
        out['K'] = K
        out['sched'] = 't_opt'
        out['t'] = t_opt
        out['seed'] = seed
        sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
        print(sdp_c, sdp_canontime, sdp_solvetime)

        # out_df = []
        out_res.append(pd.Series(out))
        out_df = pd.DataFrame(out_res)
        print(out_df)

        # out_df.to_csv('data/strongcvx/sdp_silver_m15n8_rhalf.csv', index=False)


def double_silver_experiments():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
    outf_prefix = '/home/vranjan/algorithm-certification/'
    # outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/NNLS/data/{curr_time}.csv'
    print(outf)

    # m, n = 30, 15
    m, n = 60, 40
    b_cmul = 20
    b_c = b_cmul * np.ones((m, 1))
    b_r = .5
    seed = 1

    instance = NNLS(m, n, b_c, b_r, ATA_mu=20, seed=seed)
    print(instance.A)


def silver_vs_opt():
    # m, n = 30, 15
    # m, n = 60, 40
    # bc_mul = 20
    # b_c = bc_mul * np.ones((m, 1))
    # b_r = 1
    # K_max = 10
    # seed = 1

    # instance = NNLS(m, n, b_c, b_r, seed=seed)
    # print(instance.L, instance.mu, instance.kappa)
    # print(instance.get_t_opt())
    # compute_silver_idx(instance.kappa, 16)
    # compute_silver_steps(instance.kappa, 16)
    # silver_vs_opt_glob(m, n, b_c, b_r, instance, K_max)

    # silver_vs_opt_sdp(m, n, bc_mul, b_r, instance, K_max, seed)
    double_silver_experiments()


def main():
    silver_vs_opt()


if __name__ == '__main__':
    main()
