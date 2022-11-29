import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    # "font.family": "sans-serif",   # This is needed only in the slides
    # "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_resids(df, data_dir):
    print('plotting residuals')

    N_vals = df['num_iter'].unique()
    N_vals = [1, 2, 3]
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for N in N_vals:
        df_temp = df[df['num_iter'] == N]
        xinit_eps_vals = df_temp['xinit_eps'].to_numpy()
        times = df_temp['global_solve_time'].to_numpy()
        cvars = df_temp['global_cvar'].to_numpy()
        ax[0].plot(xinit_eps_vals, -cvars, label=f'K={N}')
        ax[1].plot(xinit_eps_vals, times, label=f'K={N}')

    ax[0].set_title('In sample CVar loss')
    ax[0].set_xlabel(r'$\varepsilon$')
    ax[0].set_xscale('log')
    # plt.ylabel('in sample CVar')
    ax[0].set_yscale('symlog')

    ax[1].set_title('Time(s)')
    ax[1].set_xlabel(r'$\varepsilon$')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    #
    plt.legend()
    plt.savefig('quadcopter_data/images/qc_cvar_times.pdf')
    plt.show()


def plot_times(df, data_dir):
    print('plotting residuals')

    N_vals = df['num_iter'].unique()
    N_vals = [1, 2, 3]
    fig, ax = plt.subplots(figsize=(6, 4))
    for N in N_vals:
        df_temp = df[df['num_iter'] == N]
        xinit_eps_vals = df_temp['xinit_eps'].to_numpy()
        times = df_temp['global_solve_time'].to_numpy()
        ax.plot(xinit_eps_vals, times, label=f'N={N}')

    plt.title('Solve time (s)')
    plt.xlabel(r'$\varepsilon$')
    plt.xscale('log')
    # plt.ylabel('times')
    plt.yscale('log')
    #
    plt.legend()
    plt.savefig('quadcopter_data/images/qc_times.pdf')
    plt.show()


def main():
    # data_dir = '/home/vranjan/algorithm-certification/experiments/control/data/'
    data_dir = \
        '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/control/quadcopter_data/'
    fname = data_dir + 'qc_N1.csv'
    # plot_times(df)
    df = pd.read_csv(fname)
    print(df)
    plot_resids(df, data_dir)
    # plot_times(df, data_dir)


if __name__ == '__main__':
    main()
