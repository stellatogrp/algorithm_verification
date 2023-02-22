import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_resids(df, data_dir):
    rho_vals = df['rho'].unique()
    K_vals = sorted(df['num_iter'].unique())
    colors = ['b', 'r', 'g', 'm', 'y']

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, rho in enumerate(rho_vals):
        # temp_dfg_obj = temp_dfg['obj'].to_numpy()
        # ax.plot(K_vals, temp_dfs_obj, label='sdp', color='b')
        # temp_dfs = df_s[df_s['eps_b'] == eps]
        temp_df = df[df['rho'] == rho]
        g_df_obj = temp_df['global_res'].to_numpy()
        sdp_df_obj = temp_df['sdp_res'].to_numpy()

        ax.plot(K_vals, g_df_obj, label=f'{np.round(rho, 2)}', color=colors[i], linestyle='solid')
        ax.plot(K_vals, sdp_df_obj, color=colors[i], linestyle='dashed')

    plt.yscale('log')
    plt.ylabel(r'maximum $|| z^K - z^{K-1} ||_2^2$')

    plt.xlabel('K')

    plt.title('UQP example, eps=.05, opt_rho=16.01')
    plt.legend()
    # plt.show()
    plt.savefig(data_dir + 'resids.pdf')


def plot_times(df, data_dir):
    rho_vals = df['rho'].unique()
    K_vals = sorted(df['num_iter'].unique())
    colors = ['b', 'r', 'g', 'm', 'y']

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, rho in enumerate(rho_vals):
        # temp_dfg_obj = temp_dfg['obj'].to_numpy()
        # ax.plot(K_vals, temp_dfs_obj, label='sdp', color='b')
        # temp_dfs = df_s[df_s['eps_b'] == eps]
        temp_df = df[df['rho'] == rho]
        g_df_obj = temp_df['global_comp_time'].to_numpy()
        sdp_df_obj = temp_df['sdp_comp_time'].to_numpy()

        ax.plot(K_vals, 10 * g_df_obj, label=f'{np.round(rho, 2)}', color=colors[i], linestyle='solid')
        ax.plot(K_vals, 10 * sdp_df_obj, color=colors[i], linestyle='dashed')

    plt.yscale('log')
    plt.ylabel(r'maximum $|| z^K - z^{K-1} ||_2^2$')

    plt.xlabel('K')

    plt.title('UQP example, eps=.05, opt_rho=16.01')
    plt.legend()
    # plt.show()
    plt.savefig(data_dir + 'times.pdf')


def main():
    np.random.seed(0)
    data_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/UQP/data/PSD_UQP/'
    fname = data_dir + 'samples.csv'
    df = pd.read_csv(fname)
    print(df)
    plot_resids(df, data_dir)
    plot_times(df, data_dir)


if __name__ == '__main__':
    main()
