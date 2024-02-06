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
    m, n = 10, 15
    b_cmul = 10
    b_c = b_cmul * np.ones((m, 1))
    b_r = .25
    lambd = 10
    t = .04
    seed = 3
    # K = 5
    K_vals = [4]
    # K_vals = [9, 10]
    # K_vals = [6, 7]
    # K_vals = [5, 6, 7]
    # K_vals = [1, 2, 3, 4]
    # K_vals = [1]

    instance = ISTA(m, n, b_c, b_r, lambd=lambd, seed=seed)

    # t_vals = list(instance.get_t_vals())
    # t_vals = generate_all_t_vals(t_vals)

    out_res = []
    algs = ['ista', 'fista']
    # algs = ['fista']
    for K in K_vals:
        for alg in algs:
            if alg == 'ista':
                CP = instance.generate_CP(K, t=t)
            if alg == 'fista':
                CP = instance.generate_FISTA_CP(K, t=t)
            glob = True
            if not glob:
                out = CP.solve(solver_type='SDP_CUSTOM')
                sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
                print(sdp_c, sdp_canontime, sdp_solvetime)
            else:
                out = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600)

            out['orig_m'] = m
            out['orig_n'] = n
            out['b_cmul'] = b_cmul
            out['b_r'] = b_r
            out['K'] = K
            out['t'] = t
            out['alg'] = alg
            out['lambd'] = lambd
            out['seed'] = seed
            print(out)
            out_res.append(pd.Series(out))
            out_df = pd.DataFrame(out_res)
            print(out_df)
            out_df.to_csv(outf, index=False)


if __name__ == '__main__':
    main()
