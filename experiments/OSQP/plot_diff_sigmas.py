import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_diff_sigmas(df):
    print(df)

    sigma_vals = df['sigma'].unique()
    fig, ax = plt.subplots(figsize=(6, 4))

    for sigma in sigma_vals:
        tempdf = df[df['sigma'] == sigma]
        tempdfobj = tempdf['obj'].to_numpy()
        N_vals = tempdf['num_iter'].to_numpy()
        ax.plot(N_vals, tempdfobj, label=f'sigma={sigma}')

    plt.yscale('log')
    plt.ylabel('maximum convergence residual')

    plt.xlabel('K')

    plt.title('OSQP with varying sigma values')
    plt.legend()
    plt.show()


def main():
    # data_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/OSQP/data/'
    # fname = data_dir + 'test_OSQP_data.csv'
    fname = "data/diff_sigma_vals.csv"
    df = pd.read_csv(fname)
    #  df_pep = pd.read_csv(pep_fname)
    #  df_avg = pd.read_csv(avg_fname)
    # print(df_pep)
    # plot_vals(df)
    # plot_times(df)

    # plot_fixed_point_resids()
    plot_diff_sigmas(df)


if __name__ == '__main__':
    main()
