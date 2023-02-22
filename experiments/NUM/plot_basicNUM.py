import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_max(df, image_dir):
    fig, (ax0, ax1) = plt.subplots(2, figsize=(6, 8))
    K_vals = sorted(df['K'].unique())

    # max conv resid
    df_max = df[df['min_max'] == 'max']

    # ax.plot(K_vals, g_df_obj, label=f'{np.round(rho, 2)}', color=colors[i], linestyle='solid')
    # ax.plot(K_vals, sdp_df_obj, color=colors[i], linestyle='dashed')

    g_obj = df_max['g_obj'].to_numpy()
    sdp_obj = df_max['sdp_obj'].to_numpy()
    ax0.plot(K_vals, g_obj, color='b', linestyle='solid')
    ax0.plot(K_vals, sdp_obj, color='b', linestyle='dashed')
    ax0.set_yscale('log')
    ax0.set_ylabel('maximum convergence residual')

    g_times = df_max['g_solve_time'].to_numpy()
    sdp_times = df_max['sdp_solve_time'].to_numpy()
    ax1.plot(K_vals, g_times, color='b', linestyle='solid')
    ax1.plot(K_vals, sdp_times, color='b', linestyle='dashed')
    ax1.set_yscale('log')
    ax1.set_ylabel('solve times(s)')

    ax1.set_xlabel('K')

    fig.suptitle('NUM Upper Bounds')

    # plt.show()
    plt.savefig('images/basicNUM_max.pdf')


def plot_min(df, image_dir):
    fig, (ax0, ax1) = plt.subplots(2, figsize=(6, 8))
    K_vals = sorted(df['K'].unique())

    # max conv resid
    df_min = df[df['min_max'] == 'min']

    # ax.plot(K_vals, g_df_obj, label=f'{np.round(rho, 2)}', color=colors[i], linestyle='solid')
    # ax.plot(K_vals, sdp_df_obj, color=colors[i], linestyle='dashed')

    g_obj = df_min['g_obj'].to_numpy()
    sdp_obj = df_min['sdp_obj'].to_numpy()
    ax0.plot(K_vals, g_obj, color='g', linestyle='solid')
    ax0.plot(K_vals, sdp_obj, color='g', linestyle='dashed')
    ax0.set_yscale('log')
    ax0.set_ylabel('minimum convergence residual')

    g_times = df_min['g_solve_time'].to_numpy()
    sdp_times = df_min['sdp_solve_time'].to_numpy()
    ax1.plot(K_vals, g_times, color='g', linestyle='solid')
    ax1.plot(K_vals, sdp_times, color='g', linestyle='dashed')
    ax1.set_yscale('log')
    ax1.set_ylabel('solve times(s)')

    ax1.set_xlabel('K')

    fig.suptitle('NUM Lower Bounds')

    # plt.show()
    plt.savefig('images/basicNUM_min.pdf')


def main():
    data_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/NUM/'
    df = pd.read_csv(data_dir + 'data/basic.csv')
    image_dir = data_dir + 'images/'
    print(df)
    plot_max(df, image_dir)
    plot_min(df, image_dir)


if __name__ == '__main__':
    main()
