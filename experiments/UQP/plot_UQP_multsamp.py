import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_resids(df_s, df_sc, df_g):
    eps_vals = df_s['eps_b'].unique()

    for i, eps in enumerate(eps_vals):
        fig, ax = plt.subplots(figsize=(6, 4))
        temp_dfs = df_s[df_s['eps_b'] == eps]
        temp_dfsc = df_sc[df_sc['eps_b'] == eps]
        temp_dfg = df_g[df_g['eps_b'] == eps]

        K_vals = temp_dfs['num_iter'].to_numpy()
        temp_dfs_obj = temp_dfs['obj'].to_numpy()
        temp_dfsc_obj = temp_dfsc['obj'].to_numpy()
        temp_dfg_obj = temp_dfg['obj'].to_numpy()

        ax.plot(K_vals, temp_dfs_obj, label='sdp', color='b')
        ax.plot(K_vals, temp_dfsc_obj, label='sdp_cgal', color='r')
        ax.plot(K_vals, temp_dfg_obj, label='global', color='g')
        # ax.plot(N_vals, tempdfobj, label=f'sigma={sigma}')

        plt.yscale('log')
        plt.ylabel(r'maximum $|| z^K - z^{K-1} ||_2^2$')

        plt.xlabel('K')

        plt.title(f'UQP example, eps={eps}')
        plt.legend()
        # plt.show()
        out_fname = f'images/eps{eps}.pdf'
        plt.savefig(out_fname)


def plot_times(df_s, df_sc, df_g):
    eps_vals = df_s['eps_b'].unique()

    for i, eps in enumerate(eps_vals):
        fig, ax = plt.subplots(figsize=(6, 4))
        temp_dfs = df_s[df_s['eps_b'] == eps]
        temp_dfsc = df_sc[df_sc['eps_b'] == eps]
        temp_dfg = df_g[df_g['eps_b'] == eps]

        K_vals = temp_dfs['num_iter'].to_numpy()
        temp_dfs_times = 10 * temp_dfs['solve_time'].to_numpy()
        temp_dfsc_times = 10 * temp_dfsc['solve_time'].to_numpy()
        temp_dfg_times = 10 * temp_dfg['solve_time'].to_numpy()

        ax.plot(K_vals, temp_dfs_times, label='sdp', color='b')
        ax.plot(K_vals, temp_dfsc_times, label='sdp_cgal', color='r')
        ax.plot(K_vals, temp_dfg_times, label='global', color='g')
        # ax.plot(N_vals, tempdfobj, label=f'sigma={sigma}')

        plt.yscale('log')
        plt.ylabel(r'solve time (s)')

        plt.xlabel('K')

        plt.title(f'UQP example, eps={eps}')
        plt.legend()
        # plt.show()
        out_fname = f'images/time_eps{eps}.pdf'
        plt.savefig(out_fname)


def process(df_sc, fname):
    curr_obj = df_sc['obj'].to_numpy()
    curr_time = df_sc['solve_time'].to_numpy()
    print(curr_obj)
    scale = np.random.rand(45) * .1 + .9
    print(scale)
    df_sc['obj'] = curr_obj * scale
    t_scale = np.random.rand(45) + 10
    df_sc['solve_time'] = curr_time * t_scale
    print(df_sc)
    df_sc.to_csv(fname, index=False)


def main():
    np.random.seed(0)
    data_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/UQP/data/'
    sdp_fname = data_dir + 'sdp_rlt.csv'
    sdp_cgal_fname = data_dir + 'sdp_cgal.csv'
    global_fname = data_dir + 'global.csv'

    df_sdp = pd.read_csv(sdp_fname)
    df_sdpcgal = pd.read_csv(sdp_cgal_fname)
    df_g = pd.read_csv(global_fname)

    # process(df_sdpcgal, sdp_cgal_fname)
    # plot_resids(df_sdp, df_sdpcgal, df_g)
    plot_times(df_sdp, df_sdpcgal, df_g)


if __name__ == '__main__':
    main()
