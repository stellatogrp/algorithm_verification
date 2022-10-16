import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_resids(df, df_ws, df_avg, df_pep):
    print('plotting residuals')
    N_vals = df['num_iter'].to_numpy()
    resid_vals = df['global_res'].to_numpy()
    ws_resid_vals = df_ws['global_res'].to_numpy()
    avg_resid_vals = df_avg['cs_resid_avg'].to_numpy()
    avg_ws_resid_vals = df_avg['ws_resid_avg'].to_numpy()
    pep_N_vals = df_pep['num_iter'].to_numpy()
    pep_vals = df_pep['pep_val'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_vals, resid_vals, label='QCQP', color='red')
    ax.plot(N_vals, ws_resid_vals, label=' Warm Start QCQP', color='blue')
    ax.plot(N_vals, avg_resid_vals, label='Sample avg', color='purple', linestyle='--')
    ax.plot(N_vals, avg_ws_resid_vals, label='Warm started sample avg', color='orange', linestyle='--')
    ax.plot(pep_N_vals, pep_vals, label='Theoretical bound', color='green')

    plt.title('Fixed point residiuals, control example')
    plt.xlabel('$N$')
    plt.ylabel('maximum fixed point residual')
    plt.yscale('log')
    #
    plt.legend()
    # plt.show()
    plt.savefig('images/fixedpoint_N9.pdf')


def plot_times(df, df_ws):
    print('plotting times')
    N_vals = df['num_iter'].to_numpy()
    resid_times = df['global_comp_time'].to_numpy()
    ws_times = df_ws['global_comp_time'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_vals, resid_times, label='QCQP', color='red')
    ax.plot(N_vals, ws_times, label='Warm Start QCQP', color='blue')

    plt.title('Fixed point residuals, control example')
    plt.xlabel('$N$')
    plt.ylabel('computation time')
    plt.yscale('log')
    #
    plt.legend()
    # plt.show()
    plt.savefig('images/fixedpoint_N9_times.pdf')


def main():
    data_dir = '/home/vranjan/algorithm-certification/experiments/control/data/'
    fname = data_dir + 'testN9.csv'
    ws_fname = data_dir + 'testWSN9.csv'
    avg_fname = data_dir + 'avg_N9.csv'
    pep_fname = data_dir + 'pep_N9.csv'
    df = pd.read_csv(fname)
    df_ws = pd.read_csv(ws_fname)
    df_avg = pd.read_csv(avg_fname)
    df_pep = pd.read_csv(pep_fname)
    # plot_times(df)
    print(df)
    plot_resids(df, df_ws, df_avg, df_pep)
    plot_times(df, df_ws)


if __name__ == '__main__':
    main()
