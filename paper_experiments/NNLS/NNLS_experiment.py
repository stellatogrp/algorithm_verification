from datetime import datetime

import numpy as np
import pandas as pd
from NNLS_class import NNLS


def generate_all_t_vals(t_vals, num_between=2):
    t_min, t_opt, t_max = t_vals
    # t_min_to_opt = np.logspace(np.log10(t_min), np.log10(t_opt), num=num_between+1)
    # t_opt_to_max = np.logspace(np.log10(t_opt), np.log10(t_max), num=num_between+1)

    t_min_to_opt = np.linspace(t_min, t_opt, num=num_between+1)
    t_opt_to_max = np.linspace(t_opt, t_max, num=num_between+1)
    # print(t_min_to_opt)
    # print(t_opt_to_max)
    t_out = np.hstack([t_min_to_opt, t_opt_to_max[1:]])
    print(t_out)
    return t_out


def main():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
    outf_prefix = '/home/vranjan/algorithm-certification/'
    # outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/NNLS/data/{curr_time}.csv'
    print(outf)

    # m, n = 30, 15
    m, n = 60, 40
    b_cmul = 30
    b_c = b_cmul * np.ones((m, 1))
    b_c[30:] = 0
    b_r = .5
    # K = 5
    # K_vals = [9]
    # K_vals = [10]
    K_vals = [7, 8]
    # K_vals = [1, 2, 3, 4, 5, 6]
    K_vals = [1]

    instance = NNLS(m, n, b_c, b_r, ATA_mu=20, seed=1)
    print(instance.mu, instance.L, instance.kappa)
    print(instance.A)

    # t_vals = list(instance.get_t_vals())
    # print('t_min, t_opt, t_max:', t_vals)
    # t_vals = generate_all_t_vals(t_vals)
    # print('t_values:', t_vals)
    # exit(0)
    t_vals = np.array(instance.grid_t_vals())
    print('t_values:', t_vals)
    t_vals = t_vals[3:4]
    # exit(0)

    out_res = []
    for K in K_vals:
        for t in t_vals:
        # t = t_vals[1]
            CP = instance.generate_CP(t, K)
            out = CP.solve(solver_type='SDP_CUSTOM')
            # out = CP.solve(solver_type='GLOBAL', add_bounds=True)
            out['orig_m'] = m
            out['orig_n'] = n
            out['b_r'] = b_r
            out['b_c'] = b_cmul
            out['K'] = K
            out['t'] = t
            sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
            # del out['primal_sol']
            print(sdp_c, sdp_canontime, sdp_solvetime)

            # out_df = []
            out_res.append(pd.Series(out))
            out_df = pd.DataFrame(out_res)
            print(out_df)
            # out_df.to_csv(outf, index=False)


if __name__ == '__main__':
    main()
