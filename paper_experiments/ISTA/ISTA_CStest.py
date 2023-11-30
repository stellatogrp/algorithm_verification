from datetime import datetime

import numpy as np
import pandas as pd
from ISTA_class import ISTA


def rank1(X):
    U, sigma, VT = np.linalg.svd(X, full_matrices=False)
    return sigma[0] * np.outer(U[:, 0], U[:, 0])


def main():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
    # outf_prefix = '/home/vranjan/algorithm-certification/'
    outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/ISTA/data/{curr_time}.csv'
    print(outf)

    # m, n = 30, 15
    m, n = 5, 3
    b_c = 10 * np.ones((m, 1))
    b_r = 1
    # K = 5
    K_vals = [3]
    # K_vals = [9, 10]
    # K_vals = [7, 8]
    # K_vals = [1, 2, 3, 4, 5, 6]

    instance = ISTA(m, n, b_c, b_r, lambd=0.01, seed=1)

    # t_vals = list(instance.get_t_vals())
    # t_vals = generate_all_t_vals(t_vals)

    out_res = []
    for K in K_vals:
        # for t in t_vals:
        # t = t_vals[1]
        t = .05
        # CP = instance.generate_CP(K, t=t)
        CP = instance.generate_FISTA_CP(K, t=t)
        glob = False
        if not glob:
            out = CP.solve(solver_type='SDP_CUSTOM')
            out['orig_m'] = m
            out['orig_n'] = n
            out['b_r'] = b_r
            out['K'] = K
            out['t'] = t
            sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
            print(sdp_c, sdp_canontime, sdp_solvetime)

            # out_df = []
            out_res.append(pd.Series(out))
            out_df = pd.DataFrame(out_res)
            print(out_df)
            exit(0)
            # out_df.to_csv(outf, index=False)
            x_sol = out['primal_sol']
            print(x_sol.shape)
            iter_range_map = CP.solver.handler.iter_bound_map
            iter_list = CP.solver.handler.iterate_list
            print(iter_list)
            z = iter_list[3]
            print(z)
            print(iter_range_map[z])
            zKrange = iter_range_map[z][K]
            zKm1range = iter_range_map[z][K-1]
            print(zKrange, zKm1range)
            zK = x_sol[-1, zKrange[0]:zKrange[1]]
            zKm1 = x_sol[-1, zKm1range[0]:zKm1range[1]]
            print(np.linalg.norm(zK - zKm1) ** 2)

            zK_zKT = x_sol[zKrange[0]:zKrange[1], zKrange[0]:zKrange[1]]
            zKm1_zKm1T = x_sol[zKm1range[0]:zKm1range[1], zKm1range[0]:zKm1range[1]]
            zK_zkM1T = x_sol[zKrange[0]:zKrange[1], zKm1range[0]:zKm1range[1]]

            print(np.trace(zK_zKT - 2 * zK_zkM1T + zKm1_zKm1T))
            print(2 * np.trace(zK_zkM1T))
            print(np.trace(zK_zKT + zKm1_zKm1T))

            Xr1 = rank1(x_sol)
            print(Xr1.shape)

            print('rank 1:')
            zK_zKT = Xr1[zKrange[0]:zKrange[1], zKrange[0]:zKrange[1]]
            zKm1_zKm1T = Xr1[zKm1range[0]:zKm1range[1], zKm1range[0]:zKm1range[1]]
            zK_zkM1T = Xr1[zKrange[0]:zKrange[1], zKm1range[0]:zKm1range[1]]

            print(np.trace(zK_zKT - 2 * zK_zkM1T + zKm1_zKm1T))
            print(2 * np.trace(zK_zkM1T))
            print(np.trace(zK_zKT + zKm1_zKm1T))

            print('check DD:')
            print(2 * np.diag(x_sol) - np.sum(np.abs(x_sol), axis=1))

        else:
            out = CP.solve(solver_type='GLOBAL', add_bounds=True)
            print(out)


if __name__ == '__main__':
    main()
