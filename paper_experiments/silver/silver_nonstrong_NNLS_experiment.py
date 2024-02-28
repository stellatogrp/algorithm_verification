from datetime import datetime

import numpy as np
import pandas as pd
from NNLS import NNLS


def silver_nonstrong_cvx_only():
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
    K_max = 10

    instance = NNLS(m, n, b_c, b_r, ATA_mu=0, seed=seed)
    print(instance.A)

    K_vals = [1, 2, 3, 4, 5, 6, 7]
    silvers = instance.get_silver_steps(K_max)

    t = 1.5 / instance. L
    t_vals = [t] * K_max

    out_res = []
    for K in K_vals:
        i = K - 1
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

        out_res.append(pd.Series(out))
        out_df = pd.DataFrame(out_res)
        print(out_df)
        out_df.to_csv(outf, index=False)

        CP = instance.generate_CP(t_vals, K)
        out = CP.solve(solver_type='SDP_CUSTOM')
        out['orig_m'] = m
        out['orig_n'] = n
        out['mu'] = instance.mu
        out['L'] = instance.L
        out['b_cmul'] = b_cmul
        out['b_r'] = b_r
        out['K'] = K
        out['sched'] = 'fixed'
        out['t'] = t_vals[i]
        out['seed'] = seed
        sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
        print(sdp_c, sdp_canontime, sdp_solvetime)

        out_res.append(pd.Series(out))
        out_df = pd.DataFrame(out_res)
        print(out_df)
        out_df.to_csv(outf, index=False)


def main():
    silver_nonstrong_cvx_only()


if __name__ == '__main__':
    main()
