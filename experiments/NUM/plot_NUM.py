import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_res_times(df, plot_fname, title):
    print(df)

    K_vals = df['K'].to_numpy()
    fig, (ax0, ax1) = plt.subplots(2, figsize=(6, 8))

    ax0.plot(K_vals, df['sdp'], label='sdp')
    ax0.plot(K_vals, df['sdp_r'], label='sdp_rlt')
    ax0.plot(K_vals, df['sdp_p'], label='sdp_rlt_planet')
    ax0.plot(K_vals, df['glob'], label='global')

    ax1.plot(K_vals, df['sdptime'], label='sdp')
    ax1.plot(K_vals, df['sdp_rtime'], label='sdp_rlt')
    ax1.plot(K_vals, df['sdp_ptime'], label='sdp_rlt_planet')
    ax1.plot(K_vals, df['glob_time'], label='global')

    ax0.set_ylabel('obj vals')
    ax1.set_ylabel('times (s)')

    ax0.set_yscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('K')

    ax0.legend()
    fig.suptitle(title)
    plt.savefig(plot_fname)
    plt.show()


def main():
    data_fname1 = \
        '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/NUM/data/test_no_cross.csv'
    plot_fname1 = \
        '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/NUM/data/test_no_cross.pdf'
    title1 = 'small NUM example, no linstep cross'

    data_fname2 = \
        '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/NUM/data/test_cross.csv'
    plot_fname2 = \
        '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/NUM/data/test_cross.pdf'
    title2 = 'small NUM example with linstep cross'

    df1 = pd.read_csv(data_fname1)
    plot_res_times(df1, plot_fname1, title1)

    df2 = pd.read_csv(data_fname2)
    plot_res_times(df2, plot_fname2, title2)


if __name__ == '__main__':
    main()
