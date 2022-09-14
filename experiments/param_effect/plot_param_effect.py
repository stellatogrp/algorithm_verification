import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot_all_points(df_samples, df_pep):
    samples_n_vals = df_samples['n'].to_numpy()
    samples_resid_vals = df_samples['conv_resid'].to_numpy()
    samples_warm_start_resid_vals = df_samples['warm_start_conv_resid'].to_numpy()
    pep_n_vals = df_pep['n'].to_numpy()
    pep_resid_vals = df_pep['pepit_max_sample_obj'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(samples_n_vals, samples_resid_vals, label='samples', s=4)
    ax.scatter(samples_n_vals, samples_warm_start_resid_vals, label='warm start samples', s=4)
    ax.scatter(pep_n_vals, pep_resid_vals, label='sample PEP upper bound', s=10)

    plt.title('Convergence residuals')
    plt.xlabel('$n$')
    plt.ylabel('maximum $||x^k - x^{k-1}||_2^2$')
    plt.yscale('log')

    plt.legend()
    # plt.show()
    plt.savefig('experiments/param_effect/images/1000samples_allpoints.pdf')


def plot_averages(df_samples, df_pep):
    df_avg = df_samples.groupby('n')['conv_resid'].mean()
    df_ws_avg = df_samples.groupby('n')['warm_start_conv_resid'].mean()
    print(df_avg, df_ws_avg)
    n_vals = df_pep['n'].to_numpy()
    pep_resid_vals = df_pep['pepit_max_sample_obj'].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 4))
    # ax.scatter(samples_n_vals, samples_resid_vals, label='samples', s=4)
    # ax.scatter(samples_n_vals, samples_warm_start_resid_vals, label='warm start samples', s=4)
    # ax.scatter(pep_n_vals, pep_resid_vals, label='sample PEP upper bound', s=10)
    ax.plot(n_vals, df_avg.to_numpy(), label='Sample avg', color='blue', linestyle='--')
    ax.plot(n_vals, df_ws_avg.to_numpy(), label='Warm started sample avg', color='orange', linestyle='--')
    ax.plot(n_vals, pep_resid_vals, label='Theoretical bound', color='green')

    plt.title(r'Average convergence residuals, $k=5, \mu=1, L=10$')
    plt.xlabel('$n$')
    plt.ylabel('maximum $||x^k - x^{k-1}||_2^2$')
    plt.yscale('log')
    # plt.xscale('log')

    plt.legend()
    # plt.show()
    plt.savefig('experiments/param_effect/images/100samples_averages.pdf')


def main():
    data_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/param_effect/data/'
    sample_fname = data_dir + 'test_outb13sample100.csv'
    pep_fname = data_dir + 'test_pepb13sample100.csv'
    df_samples = pd.read_csv(sample_fname)
    df_pep = pd.read_csv(pep_fname)
    # print(df_samples)
    # plot_all_points(df_samples, df_pep)
    plot_averages(df_samples, df_pep)


if __name__ == '__main__':
    main()
