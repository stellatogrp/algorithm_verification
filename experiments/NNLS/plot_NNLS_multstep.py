import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
    })
import pandas as pd


def plot_vals(df, df_pep, df_avg):

    N_vals = df['num_iter'].to_numpy()
    resid_vals = df['conv_resid'].to_numpy()
    ws_resid_vals = df['warm_start_conv_resid'].to_numpy()
    pep_bound_vals = df_pep['pep_bound'].to_numpy()

    # df_samples.groupby('n')['conv_resid'].mean()
    # avg_resid_vals = df_avg.groupby('max_N')['res'].mean().to_numpy()
    # avg_ws_resid_vals = df_avg.groupby('max_N')['ws_res'].mean().to_numpy()
    avg_resid_vals = df_avg['avg_res'].to_numpy()
    avg_ws_resid_vals = df_avg['ws_avg_res'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_vals, resid_vals, label='QCQP', color='red')
    ax.plot(N_vals, ws_resid_vals, label='Warm started QCQP', color='purple')
    ax.plot(N_vals, pep_bound_vals, label='Theoretical bound', color='green')
    ax.plot(N_vals, avg_resid_vals, label='Sample avg', color='blue', linestyle='--')
    ax.plot(N_vals, avg_ws_resid_vals, label='Warm started sample avg', color='orange', linestyle='--')

    plt.title('Convergence residuals')
    plt.xlabel('$N$')
    plt.ylabel('maximum $||x^N - x^{N-1}||_2^2$')
    plt.yscale('log')
    #
    plt.legend()
    # plt.show()
    plt.savefig('images/NNLS_multstep.pdf')


def plot_times(df):
    N_vals = df['num_iter'].to_numpy()
    resid_vals_time = df['conv_resid_comp_time'].to_numpy()
    ws_resid_vals_time = df['warm_start_conv_resid_comp_time'].to_numpy()

    print(resid_vals_time, ws_resid_vals_time)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_vals, resid_vals_time, label='QCQP', color='red')
    ax.plot(N_vals, ws_resid_vals_time, label='Warm started QCQP', color='purple')

    plt.title('Solve times')
    plt.xlabel('$N$')
    plt.ylabel('time (s)')
    plt.yscale('log')
    #
    plt.legend()
    # plt.show()
    plt.savefig('images/NNLS_multstep_times.pdf')


def main():
    data_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/NNLS/data/'
    fname = data_dir + 'test_NNLS_multstep.csv'
    pep_fname = data_dir + 'test_NNLS_multstep_PEP.csv'
    avg_fname = data_dir + 'test_NNLS_multstep_avg_alttest.csv'
    df = pd.read_csv(fname)
    df_pep = pd.read_csv(pep_fname)
    df_avg = pd.read_csv(avg_fname)
    # print(df_pep)
    # plot_vals(df, df_pep, df_avg)
    plot_times(df)


if __name__ == '__main__':
    main()
