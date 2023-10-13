from datetime import datetime

import numpy as np
<<<<<<< HEAD
from NNLS_class import NNLS


def generate_all_t_vals(t_vals, num_between=1):
=======
import pandas as pd
from NNLS_class import NNLS


def generate_all_t_vals(t_vals, num_between=3):
>>>>>>> 7f95bb1 (updated experiments with lower mosek tol)
    t_min, t_opt, t_max = t_vals
    t_min_to_opt = np.logspace(np.log10(t_min), np.log10(t_opt), num=num_between+1)
    t_opt_to_max = np.logspace(np.log10(t_opt), np.log10(t_max), num=num_between+1)
    # print(t_min_to_opt)
    # print(t_opt_to_max)
    t_out = np.hstack([t_min_to_opt, t_opt_to_max[1:]])
    print(t_out)
    return t_out


def main():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
<<<<<<< HEAD
    # outf_prefix = '/home/vranjan/algorithm-certification/'
    outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/NNLS/data/{curr_time}.csv'
=======
    outf_prefix = '/home/vranjan/algorithm-certification/'
    # outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/NNLS/data/glob_{curr_time}.csv'
>>>>>>> 7f95bb1 (updated experiments with lower mosek tol)
    print(outf)

    m, n = 30, 15
    b_c = 10 * np.ones((m, 1))
    b_r = .1
    # K = 5
    # K_vals = [1, 2, 3, 4, 6]
<<<<<<< HEAD
    K_vals = [1]
=======
    # K_vals = [1]
>>>>>>> 7f95bb1 (updated experiments with lower mosek tol)

    instance = NNLS(m, n, b_c, b_r, seed=1)

    t_vals = list(instance.get_t_vals())
    t_vals = generate_all_t_vals(t_vals)
<<<<<<< HEAD
    print(t_vals)

    K_vals = [6]
    t_vals = t_vals[:1]
=======

    out_res = []
    K_vals = [6]
>>>>>>> 7f95bb1 (updated experiments with lower mosek tol)
    print(t_vals)
    for K in K_vals:
        for t in t_vals:
            print(t, K)
            CP = instance.generate_CP(t, K)
<<<<<<< HEAD
            CP.solve(solver_type='GLOBAL', add_bounds=True)
    # for K in K_vals:
    #     for t in t_vals:
    #     # t = t_vals[1]
    #         CP = instance.generate_CP(t, K)
=======
            out_g, out_time = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600)
            out_dict = dict(glob_objval=out_g, glob_solvetime=out_time, orig_m=m, orig_n=n, K=K, t=t)
            out_res.append(pd.Series(out_dict))

            out_df = pd.DataFrame(out_res)
            out_df.to_csv(outf, index=False)
    # for K in K_vals:
    #     for t in t_vals:
    #     # t = t_vals[1]
    #         CP = instance.generate_CP(t, K)x
>>>>>>> 7f95bb1 (updated experiments with lower mosek tol)
    #         out = CP.solve(solver_type='SDP_CUSTOM')
    #         out['orig_m'] = m
    #         out['orig_n'] = n
    #         out['b_r'] = b_r
    #         out['K'] = K
    #         out['t'] = t
    #         sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
    #         print(sdp_c, sdp_canontime, sdp_solvetime)

    #         # out_df = []
    #         out_res.append(pd.Series(out))
    #         out_df = pd.DataFrame(out_res)
    #         print(out_df)
            # out_df.to_csv(outf, index=False)


if __name__ == '__main__':
    main()
