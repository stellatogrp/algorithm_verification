import numpy as np
import pandas as pd
from NNLS import NNLS


def silver_vs_opt_sdp(m, n, b_c, b_r, instance, K_max):
    K_vals = range(1, K_max + 1)
    # K_vals = range(K_max, K_max + 1)

    silvers = instance.get_silver_steps(K_max)
    print(silvers)
    t_opt = instance.get_t_opt()
    print(t_opt)

    out_res = []
    for K in K_vals:
        CP = instance.generate_CP(silvers, K)
        out = CP.solve(solver_type='SDP_CUSTOM')
        out['orig_m'] = m
        out['orig_n'] = n
        out['mu'] = instance.mu
        out['L'] = instance.L
        out['b_r'] = b_r
        out['K'] = K
        out['t'] = 'silver'
        sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
        print(sdp_c, sdp_canontime, sdp_solvetime)

        # out_df = []
        out_res.append(pd.Series(out))
        out_df = pd.DataFrame(out_res)
        print(out_df)

        CP2 = instance.generate_CP(t_opt, K)
        out = CP2.solve(solver_type='SDP_CUSTOM')
        out['orig_m'] = m
        out['orig_n'] = n
        out['mu'] = instance.mu
        out['L'] = instance.L
        out['b_r'] = b_r
        out['K'] = K
        out['t'] = t_opt
        sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
        print(sdp_c, sdp_canontime, sdp_solvetime)

        # out_df = []
        out_res.append(pd.Series(out))
        out_df = pd.DataFrame(out_res)
        print(out_df)

        # out_df.to_csv('data/silver_NNLS_sdp.csv', index=False)


def silver_vs_opt_glob(m, n, b_c, b_r, instance, K_max):
    K_vals = range(1, K_max + 1)

    silvers = instance.get_silver_steps(K_max)
    print(silvers)
    t_opt = instance.get_t_opt()
    print(t_opt)

    out_res = []
    for K in K_vals:
        CP = instance.generate_CP(silvers, K)
        out_g, out_gtime = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=10)
        out = {}
        out['glob_objval'] = out_g
        out['glob_solvetime'] = out_gtime
        out['orig_m'] = m
        out['orig_n'] = n
        out['mu'] = instance.mu
        out['L'] = instance.L
        out['b_r'] = b_r
        out['K'] = K
        out['t'] = 'silver'

        # out_df = []
        out_res.append(pd.Series(out))
        out_df = pd.DataFrame(out_res)
        print(out_df)

        CP2 = instance.generate_CP(t_opt, K)
        out_g, out_gtime = CP2.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=10)
        out = {}
        out['glob_objval'] = out_g
        out['glob_solvetime'] = out_gtime
        out['orig_m'] = m
        out['orig_n'] = n
        out['mu'] = instance.mu
        out['L'] = instance.L
        out['b_r'] = b_r
        out['K'] = K
        out['t'] = t_opt

        # out_df = []
        out_res.append(pd.Series(out))
        out_df = pd.DataFrame(out_res)
        print(out_df)

        # out_df.to_csv('data/silver_NNLS_glob.csv', index=False)


def silver_vs_opt():
    # m, n = 30, 15
    m, n = 15, 8
    b_c = 10 * np.ones((m, 1))
    b_r = .25

    instance = NNLS(m, n, b_c, b_r, seed=1)
    print(instance.L, instance.mu, instance.kappa)
    # silver_vs_opt_sdp(m, n, b_c, b_r, instance, K_max)
    # silver_vs_opt_glob(m, n, b_c, b_r, instance, K_max)


def main():
    silver_vs_opt()


if __name__ == '__main__':
    main()
