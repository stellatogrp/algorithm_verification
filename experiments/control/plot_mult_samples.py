import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_resids(df, data_dir):
    print('plotting residuals')

    N_vals = df['num_iter'].unique()
    fig, ax = plt.subplots(figsize=(6, 4))
    for N in N_vals:
        df_temp = df[df['num_iter'] == N]
        xinit_eps_vals = df_temp['xinit_eps'].to_numpy()
        cvars = df_temp['global_cvar'].to_numpy()
        ax.plot(xinit_eps_vals, cvars, label=f'N={N}')

    plt.title('CVars')
    plt.xlabel('eps')
    plt.xscale('log')
    plt.ylabel('CVar')
    plt.yscale('log')
    #
    plt.legend()
    plt.show()


def plot_times(df, data_dir):
    print('plotting residuals')

    N_vals = df['num_iter'].unique()
    fig, ax = plt.subplots(figsize=(6, 4))
    for N in N_vals:
        df_temp = df[df['num_iter'] == N]
        xinit_eps_vals = df_temp['xinit_eps'].to_numpy()
        times = df_temp['global_solve_time'].to_numpy()
        ax.plot(xinit_eps_vals, times, label=f'N={N}')

    plt.title('Solve times')
    plt.xlabel('eps')
    plt.xscale('log')
    plt.ylabel('times')
    plt.yscale('log')
    #
    plt.legend()
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
