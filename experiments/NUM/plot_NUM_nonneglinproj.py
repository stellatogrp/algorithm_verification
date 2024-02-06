import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot(df, image_dir):
    fig, (ax0, ax1) = plt.subplots(2, figsize=(6, 8))
    K_vals = sorted(df['K'].unique())

    df_max = df[df['min_max'] == 'max']
    s_max_obj = df_max['s_obj'].to_numpy()
    s_max_time = df_max['s_time'].to_numpy()
    c_max_obj = df_max['c_obj'].to_numpy()
    c_max_time = df_max['c_time'].to_numpy()

    df_min = df[df['min_max'] == 'min']
    s_min_obj = df_min['s_obj'].to_numpy()
    s_min_time = df_min['s_time'].to_numpy()
    c_min_obj = df_min['c_obj'].to_numpy()
    c_min_time = df_min['c_time'].to_numpy()

    ax0.plot(K_vals, s_max_obj, color='b', linestyle='solid')
    ax0.plot(K_vals, s_min_obj, color='b', linestyle='dashed')
    ax0.plot(K_vals, c_max_obj, color='g', linestyle='solid')
    ax0.plot(K_vals, c_min_obj, color='g', linestyle='dashed')

    ax0.set_yscale('log')
    ax0.set_ylabel('convergence residuals')

    ax1.plot(K_vals, s_max_time, color='b', linestyle='solid')
    ax1.plot(K_vals, s_min_time, color='b', linestyle='dashed')
    ax1.plot(K_vals, c_max_time, color='g', linestyle='solid')
    ax1.plot(K_vals, c_min_time, color='g', linestyle='dashed')

    ax1.set_yscale('log')
    ax1.set_ylabel('times (s)')
    ax1.set_xlabel('K')

    # # max conv resid
    # df_max = df[df['min_max'] == 'max']

    # g_obj = df_max['g_obj'].to_numpy()
    # sdp_obj = df_max['sdp_obj'].to_numpy()
    # ax0.plot(K_vals, g_obj, color='b', linestyle='solid')
    # ax0.plot(K_vals, sdp_obj, color='b', linestyle='dashed')
    # ax0.set_yscale('log')
    # ax0.set_ylabel('maximum convergence residual 1-norm')

    # g_times = df_max['g_solve_time'].to_numpy()
    # sdp_times = df_max['sdp_solve_time'].to_numpy()
    # ax1.plot(K_vals, g_times, color='b', linestyle='solid')
    # ax1.plot(K_vals, sdp_times, color='b', linestyle='dashed')
    # ax1.set_yscale('log')
    # ax1.set_ylabel('solve times(s)')

    # ax1.set_xlabel('K')

    fig.suptitle('NUM, blue=full, green=condensed, solid=max, dashed=min')

    # plt.show()
    plt.savefig('images/NUM_nonneglinstep.pdf')


def main():
    data_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/NUM/'
    # df = pd.read_csv(data_dir + 'data/basic.csv')
    df = pd.read_csv(data_dir + 'data/condense_times.csv')
    image_dir = data_dir + 'images/'
    # title = r'$c(\theta) \in [0, 1/2]^n$'
    print(df)
    plot(df, image_dir)


if __name__ == '__main__':
    main()
