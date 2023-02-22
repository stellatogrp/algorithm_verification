import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_max(df, image_dir):
    fig, (ax0, ax1) = plt.subplots(2, figsize=(6, 8))
    colors = ['b', 'r', 'g', 'm', 'y']
    rho_vals = df['rho'].unique()[1:]
    K_vals = sorted(df['K'].unique())

    # max conv resid
    for i, rho in enumerate(rho_vals):
        df_rho = df[df['rho'] == rho]
        df_rhomax = df_rho[df_rho['min_max'] == 'max']
        print(df_rho)

        # ax.plot(K_vals, g_df_obj, label=f'{np.round(rho, 2)}', color=colors[i], linestyle='solid')
        # ax.plot(K_vals, sdp_df_obj, color=colors[i], linestyle='dashed')

        g_obj = df_rhomax['g_obj'].to_numpy()
        sdp_obj = df_rhomax['sdp_obj'].to_numpy()
        ax0.plot(K_vals, g_obj, label=f'{np.round(rho, 2)}', color=colors[i], linestyle='solid')
        ax0.plot(K_vals, sdp_obj, color=colors[i], linestyle='dashed')
        ax0.legend()
        ax0.set_yscale('log')
        ax0.set_ylabel('maximum convergence residual')

        g_times = df_rhomax['g_solve_time'].to_numpy()
        sdp_times = df_rhomax['sdp_solve_time'].to_numpy()
        ax1.plot(K_vals, g_times, label=f'{np.round(rho, 2)}', color=colors[i], linestyle='solid')
        ax1.plot(K_vals, sdp_times, color=colors[i], linestyle='dashed')
        ax1.set_yscale('log')
        ax1.set_ylabel('solve times(s)')

        ax1.set_xlabel('K')

    fig.suptitle('Strongly Convex QP Upper Bounds, Different rho, opt_rho = 7.83')

    # plt.show()
    plt.savefig('images/basicPDQP_max.pdf')


def plot_min(df, image_dir):
    fig, (ax0, ax1) = plt.subplots(2, figsize=(6, 8))
    colors = ['b', 'r', 'g', 'm', 'y']
    rho_vals = df['rho'].unique()[1:]
    K_vals = sorted(df['K'].unique())

    # min conv resid
    for i, rho in enumerate(rho_vals):
        df_rho = df[df['rho'] == rho]
        df_rhomax = df_rho[df_rho['min_max'] == 'min']
        print(df_rho)

        # ax.plot(K_vals, g_df_obj, label=f'{np.round(rho, 2)}', color=colors[i], linestyle='solid')
        # ax.plot(K_vals, sdp_df_obj, color=colors[i], linestyle='dashed')

        g_obj = df_rhomax['g_obj'].to_numpy()
        sdp_obj = df_rhomax['sdp_obj'].to_numpy()
        ax0.plot(K_vals, g_obj, label=f'{np.round(rho, 2)}', color=colors[i], linestyle='solid')
        ax0.plot(K_vals, sdp_obj, color=colors[i], linestyle='dashed')
        ax0.legend()
        ax0.set_yscale('log')
        ax0.set_ylabel('minimum convergence residual')

        g_times = df_rhomax['g_solve_time'].to_numpy()
        sdp_times = df_rhomax['sdp_solve_time'].to_numpy()
        ax1.plot(K_vals, g_times, label=f'{np.round(rho, 2)}', color=colors[i], linestyle='solid')
        ax1.plot(K_vals, sdp_times, color=colors[i], linestyle='dashed')
        ax1.set_yscale('log')
        ax1.set_ylabel('solve times(s)')

        ax1.set_xlabel('K')

    fig.suptitle('Strongly Convex QP Lower Bounds, Different rho, opt_rho = 7.83')

    # plt.show()
    plt.savefig('images/basicPDQP_min.pdf')


def main():
    data_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/rho_opt/'
    df = pd.read_csv(data_dir + 'data/PD_QP.csv')
    image_dir = data_dir + 'images/'
    print(df)
    plot_max(df, image_dir)
    plot_min(df, image_dir)


if __name__ == '__main__':
    main()
