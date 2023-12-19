from datetime import datetime

import numpy as np
import pandas as pd
from NUM_class import NUM


def main():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
    outf_prefix = '/home/vranjan/algorithm-certification/'
    # outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/NUM/data/{curr_time}.csv'
    print(outf)

    m, n = 10, 5
    # m, n = 2, 1
    c_mul = 10
    c_c = c_mul * np.ones((m, 1))
    c_r = .5
    # K = 5
    # K_vals = [1, 2, 3, 4]
    # K_vals = [3, 4, 5]
    K_vals = [6]
    seed = 0

    instance = NUM(m, n, c_c, c_r=c_r, seed=seed)
    # instance.test_cp_prob()
    # exit(0)
    # instance = NUM_simple(m, n, c_c, c_r=c_r, seed=3)

    out_res = []
    init_types = ['cs', 'heur', 'ws']
    # init_types = ['ws']
    for K in K_vals:
        for init_type in init_types:
            # instance.generate_CP_ball(K, warm_start=False)
            # print(instance.R)
            # exit(0)

            # CP = instance.generate_CP_ball(K, warm_start=False)
            # out = CP.solve(solver_type='SDP_CUSTOM')
            # out['orig_m'] = m
            # out['orig_n'] = n
            # out['z_dim'] = m * 3 * n
            # out['K'] = K
            # out['warm_start'] = False
            # out_res.append(pd.Series(out))
            # out_df = pd.DataFrame(out_res)
            # out_df.to_csv(outf, index=False)

            CP2 = instance.generate_CP_ball(K, init_type=init_type)
            out_ws = CP2.solve(solver_type='SDP_CUSTOM')
            # out_ws = CP2.solve(solver_type='GLOBAL', add_bounds=True)
            out_ws['orig_m'] = m
            out_ws['orig_n'] = n
            out_ws['z_dim'] = m + 3 * n
            out_ws['c_mul'] = c_mul
            out_ws['K'] = K
            # out_ws['warm_start'] = True
            out_ws['init_type'] = init_type
            out_ws['seed'] = seed
            out_res.append(pd.Series(out_ws))
            out_df = pd.DataFrame(out_res)
            print(out_df)
            out_df.to_csv(outf, index=False)


if __name__ == '__main__':
    main()
