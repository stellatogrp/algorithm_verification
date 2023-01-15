import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_resids(df_srlt, df_g):
    # sample_vals = df['num_samples'].unique()
    # fig, ax = plt.subplots(figsize=(6, 4))

    r_vals = df_srlt['r_x'].unique()
    # eps_vals = [.01, .1, .5]
    # N_vals = range(2, 7)
    colors = ['g', 'r', 'b', 'orange', 'y']

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, r in enumerate(r_vals):
        tempdf_srlt = df_srlt[df_srlt['r_x'] == r]  # df[df['num_iter'] == N]
        tempdf_g = df_g[df_g['r_x'] == r]
        # xinit_eps_vals = df_temp['xinit_eps'].to_numpy()
        # times = df_temp['global_solve_time'].to_numpy()
        # ax.plot(xinit_eps_vals, times, label=f'N={N}')
        srlt_obj = tempdf_srlt['obj'].to_numpy()
        g_obj = tempdf_g['obj'].to_numpy()
        N_vals = tempdf_srlt['num_iter'].to_numpy()
        ax.plot(N_vals, srlt_obj, label=f'r_x={r}', color=colors[i])
        ax.plot(N_vals, g_obj, color=colors[i], linestyle='dashed')

    plt.yscale('log')
    plt.ylabel('obj')

    plt.xlabel('K')

    plt.title('solid = SDP_RLT, dashed = global')
    plt.legend()
    plt.show()
    # plt.savefig('data/mult_eps_test/images/varyradius_simpleOSQP.pdf')


def main():
    # data_dir = '/home/vranjan/algorithm-certification/experiments/control/data/'
    data_dir = \
        '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/OSQP/data/mult_eps_test/'
    s_rlt_fname = data_dir + 'vary_radius_sdp_rlt.csv'
    g_fname = data_dir + 'vary_radius_global.csv'
    # plot_times(df)
    df_srlt = pd.read_csv(s_rlt_fname)
    df_g = pd.read_csv(g_fname)
    print(df_srlt)
    print(df_g)
    plot_resids(df_srlt, df_g)


if __name__ == '__main__':
    main()
