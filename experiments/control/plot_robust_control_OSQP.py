import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_resids(df):
    print('plotting residuals')
    sample_vals = df['num_samples'].unique()
    fig, ax = plt.subplots(figsize=(6, 4))
    for val in sample_vals:
        df_temp = df[df['num_samples'] == val]
        N_vals = df_temp['num_iter'].to_numpy()
        resids = df_temp['global_res'].to_numpy()
        ax.plot(N_vals, resids, label=f'samples={val}')
    # N_vals = df['num_iter'].to_numpy()
    # resid_vals = df['global_res'].to_numpy()
    # # ws_resid_vals = df_ws['global_res'].to_numpy()
    # # avg_resid_vals = df_avg['cs_resid_avg'].to_numpy()
    # # avg_ws_resid_vals = df_avg['ws_resid_avg'].to_numpy()
    # # pep_N_vals = df_pep['num_iter'].to_numpy()
    # # pep_vals = df_pep['pep_val'].to_numpy()

    # fig, ax = plt.subplots(figsize=(6, 4))
    # ax.plot(N_vals, resid_vals, label='QCQP', color='red')
    # # ax.plot(N_vals, ws_resid_vals, label=' Warm Start QCQP', color='blue')
    # # ax.plot(N_vals, avg_resid_vals, label='Sample avg', color='purple', linestyle='--')
    # # ax.plot(N_vals, avg_ws_resid_vals, label='Warm started sample avg', color='orange', linestyle='--')
    # # ax.plot(pep_N_vals, pep_vals, label='Theoretical bound', color='green')

    plt.title('Fixed point residiuals, control ex, estimate param')
    plt.xlabel('$N$')
    plt.ylabel('maximum fixed point residual')
    plt.yscale('log')
    #
    plt.legend()
    # plt.show()
    plt.savefig('images/robust_resids.pdf')


def plot_times(df):
    print('plotting times')
    sample_vals = df['num_samples'].unique()
    fig, ax = plt.subplots(figsize=(6, 4))
    for val in sample_vals:
        df_temp = df[df['num_samples'] == val]
        N_vals = df_temp['num_iter'].to_numpy()
        times = df_temp['global_comp_time'].to_numpy()
        ax.plot(N_vals, times, label=f'samples={val}')

    plt.title('Fixed point residiuals, control ex, estimate param')
    plt.xlabel('$N$')
    plt.ylabel('computation time')
    plt.yscale('log')
    #
    plt.legend()
    # plt.show()
    plt.savefig('images/robust_times.pdf')


def main():
    # data_dir = '/home/vranjan/algorithm-certification/experiments/control/data/'
    data_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/control/data/'
    fname = data_dir + 'testrobust.csv'

    df = pd.read_csv(fname)
    print(df)
    plot_resids(df)
    plot_times(df)


if __name__ == '__main__':
    main()
