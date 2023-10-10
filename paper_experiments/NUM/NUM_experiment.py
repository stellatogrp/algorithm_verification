from datetime import datetime

import numpy as np
import pandas as pd
from NUM_class import NUM


def main():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
    outf_prefix = '/home/vranjan/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/NUM/data/{curr_time}.csv'
    print(outf)

    m, n = 1, 2
    c_c = np.ones((m, 1))
    c_r = .1
    # K = 5
    # K_vals = [1, 2, 3, 4, 5]
    K_vals = [1, 2]

    instance = NUM(m, n, c_c, c_r=c_r, seed=0)

    out_res = []
    for K in K_vals:
        CP = instance.generate_CP_ball(K, warm_start=False)
        out = CP.solve(solver_type='SDP_CUSTOM')
        out['orig_m'] = m
        out['orig_n'] = n
        out['z_dim'] = m * 3 * n
        out['K'] = K
        out['warm_start'] = False
        out_res.append(pd.Series(out))
        out_df = pd.DataFrame(out_res)
        out_df.to_csv(outf, index=False)

        CP2 = instance.generate_CP_ball(K, warm_start=True)
        out_ws = CP2.solve(solver_type='SDP_CUSTOM')
        out_ws['orig_m'] = m
        out_ws['orig_n'] = n
        out_ws['z_dim'] = m * 3 * n
        out_ws['K'] = K
        out_ws['warm_start'] = True
        out_res.append(pd.Series(out_ws))
        out_df = pd.DataFrame(out_res)
        out_df.to_csv(outf, index=False)


if __name__ == '__main__':
    main()
