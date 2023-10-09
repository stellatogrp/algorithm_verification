from datetime import datetime

import numpy as np
import pandas as pd
from NNLS_class import NNLS


def main():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
    outf = f'paper_experiments/NNLS/data/{curr_time}.csv'
    print(outf)

    m, n = 10, 5
    b_c = 10 * np.ones((m, 1))
    b_r = .1
    K = 1

    instance = NNLS(m, n, b_c, b_r, seed=1)

    t_vals = list(instance.get_t_vals())
    t = t_vals[1]
    CP = instance.generate_CP(t, K)
    out = CP.solve(solver_type='SDP_CUSTOM')
    out['orig_m'] = m
    out['orig_n'] = n
    out['b_r'] = b_r
    sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
    print(sdp_c, sdp_canontime, sdp_solvetime)

    out_df = []
    out_df.append(pd.Series(out))
    out_df = pd.DataFrame(out_df)
    print(pd.DataFrame(out_df))
    out_df.to_csv(outf, index=False)


if __name__ == '__main__':
    main()
