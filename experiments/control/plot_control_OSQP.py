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
    N_vals = df['num_iter'].to_numpy()
    resid_vals = df['global_res'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_vals, resid_vals, label='QCQP', color='red')

    plt.title('Convergence residuals, control example')
    plt.xlabel('$N$')
    plt.ylabel('maximum fixed point residual')
    plt.yscale('log')
    #
    plt.legend()
    # plt.show()
    plt.savefig('images/n2_fixedpoint.pdf')


def plot_times(df):
    print('plotting times')
    N_vals = df['num_iter'].to_numpy()
    resid_vals = df['global_comp_time'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_vals, resid_vals, label='QCQP', color='red')

    plt.title('Convergence residuals, control example')
    plt.xlabel('$N$')
    plt.ylabel('computation time')
    plt.yscale('log')
    #
    plt.legend()
    # plt.show()
    plt.savefig('images/n2_fixedpoint_times.pdf')


def main():
    data_dir = '/home/vranjan/algorithm-certification/experiments/control/data/'
    fname = data_dir + 'n2_fixedresid.csv'
    df = pd.read_csv(fname)
    #  df_pep = pd.read_csv(pep_fname)
    #  df_avg = pd.read_csv(avg_fname)
    # print(df_pep)
    # plot_vals(df)
    # plot_times(df)
    print(df)
    plot_resids(df)
    plot_times(df)


if __name__ == '__main__':
    main()
