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
    K_vals = sorted(df['num_iter'].unique())
    N_vals = range(1, 21)
    colors = ['b', 'r', 'g', 'm', 'y']

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, K in enumerate(K_vals):
        df_K = df[df['num_iter'] == K]
        g_plot = []
        sdp_plot = []
        for N in N_vals:
            df_sample = df_K.sample(n=N)
            g_plot.append(df_sample['global_res'].mean())
            sdp_plot.append(df_sample['sdp_res'].mean())
        ax.plot(N_vals, g_plot, label=f'K={K}', color=colors[i], linestyle='solid')
        ax.plot(N_vals, sdp_plot, color=colors[i], linestyle='dashed')

    plt.yscale('log')
    plt.ylabel(r'maximum $|| z^K - z^{K-1} ||_2^2$')

    plt.xlabel('N')

    plt.title('UQP example, eps=.05, varying N')
    plt.legend()
    # plt.show()
    plt.savefig(data_dir + 'resids.pdf')


def plot_times(df, data_dir):
    K_vals = sorted(df['num_iter'].unique())
    N_vals = range(1, 21)
    colors = ['b', 'r', 'g', 'm', 'y']

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, K in enumerate(K_vals):
        df_K = df[df['num_iter'] == K]
        g_plot = []
        sdp_plot = []
        for N in N_vals:
            df_sample = df_K.sample(n=N)
            g_plot.append(df_sample['global_comp_time'].sum())
            sdp_plot.append(df_sample['sdp_comp_time'].sum())
        ax.plot(N_vals, g_plot, label=f'K={K}', color=colors[i], linestyle='solid')
        ax.plot(N_vals, sdp_plot, color=colors[i], linestyle='dashed')

    plt.yscale('log')
    plt.ylabel(r'maximum $|| z^K - z^{K-1} ||_2^2$')

    plt.xlabel('N')

    plt.title('UQP example, eps=.05, varying N')
    plt.legend()
    # plt.show()
    plt.savefig(data_dir + 'times.pdf')


def main():
    np.random.seed(0)
    data_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/UQP/data/varyN/'
    fname = data_dir + 'samples.csv'
    df = pd.read_csv(fname)
    print(df)
    plot_resids(df, data_dir)
    plot_times(df, data_dir)


if __name__ == '__main__':
    main()
