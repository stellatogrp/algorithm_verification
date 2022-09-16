import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_vals(df):

    N_vals = df['num_iter'].to_numpy()
    sdp_resid_vals = df['conv_resid_sdp'].to_numpy()
    sdp_rlt_resid_vals = df['conv_resid_sdp_rlt'].to_numpy()
    global_resid_vals = df['conv_resid_global'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_vals, sdp_resid_vals, label='SDP', color='red')
    ax.plot(N_vals, sdp_rlt_resid_vals, label='SDP RLT', color='blue')
    ax.plot(N_vals, global_resid_vals, label='Global', color='green')

    plt.title('Convergence residuals')
    plt.xlabel('$N$')
    plt.ylabel('maximum $||x^N - x^{N-1}||_2^2$')
    plt.yscale('log')
    #
    plt.legend()
    # plt.show()
    plt.savefig('images/OSQP_resids.pdf')


def plot_times(df):
    N_vals = df['num_iter'].to_numpy()
    sdp_time = df['sdp_time'].to_numpy()
    sdp_rlt_time = df['sdp_rlt_time'].to_numpy()
    global_time = df['global_time'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_vals, sdp_time, label='SDP', color='red')
    ax.plot(N_vals, sdp_rlt_time, label='SDP RLT', color='blue')
    ax.plot(N_vals, global_time, label='Global', color='green')

    plt.title('Solve times')
    plt.xlabel('$N$')
    plt.ylabel('time (s)')
    plt.yscale('log')
    #
    plt.legend()
    # plt.show()
    plt.savefig('images/OSQP_times.pdf')


def main():
    data_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/OSQP/data/'
    fname = data_dir + 'test_OSQP_data.csv'
    df = pd.read_csv(fname)
    #  df_pep = pd.read_csv(pep_fname)
    #  df_avg = pd.read_csv(avg_fname)
    # print(df_pep)
    plot_vals(df)
    plot_times(df)
    print(df)


if __name__ == '__main__':
    main()
