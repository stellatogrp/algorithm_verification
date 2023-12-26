from datetime import datetime

import numpy as np
import pandas as pd
from ISTA_class import ISTA


def main():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
    outf_prefix = '/home/vranjan/algorithm-certification/'
    # outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/ISTA/data/{curr_time}.csv'
    print(outf)

    # m, n = 25, 20
    m, n = 20, 15
    b_cmul = 10
    b_c = b_cmul * np.ones((m, 1))
    b_r = .5
    lambd = 5
    # K = 5
    K_vals = [7]
    # K_vals = [9, 10]
    # K_vals = [6, 7]
    # K_vals = [1, 2, 3, 4, 5]

    instance = ISTA(m, n, b_c, b_r, lambd=lambd, seed=1)

    # t_vals = list(instance.get_t_vals())
    # t_vals = generate_all_t_vals(t_vals)

    out_res = []
    algs = ['ista', 'fista']
    # algs = ['fista']
    for K in K_vals:
        for alg in algs:
            # for t in t_vals:
            # t = t_vals[1]
            t = .01
            if alg == 'ista':
                CP = instance.generate_CP(K, t=t)
            if alg == 'fista':
                CP = instance.generate_FISTA_CP(K, t=t)
            glob = False
            if not glob:
                out = CP.solve(solver_type='SDP_CUSTOM')
                out['orig_m'] = m
                out['orig_n'] = n
                out['b_cmul'] = b_cmul
                out['b_r'] = b_r
                out['K'] = K
                out['t'] = t
                out['alg'] = alg
                out['lambd'] = lambd
                sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
                print(sdp_c, sdp_canontime, sdp_solvetime)

                # out_df = []
                out_res.append(pd.Series(out))
                out_df = pd.DataFrame(out_res)
                print(out_df)
                out_df.to_csv(outf, index=False)
            else:
                out = CP.solve(solver_type='GLOBAL', add_bounds=True)
                print(out)


if __name__ == '__main__':
    main()
