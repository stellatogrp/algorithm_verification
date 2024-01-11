import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def samples_to_max(samples_df, K_max=7):
    # print(samples_df)

    out = []
    for i in range(1, K_max + 1):
        ista_max = samples_df[(samples_df['K'] == i)]['ista'].max()
        fista_max = samples_df[(samples_df['K'] == i)]['fista'].max()

        out.append(pd.Series(
            {
                'K': i,
                'ista': ista_max,
                'fista': fista_max,
            }
        ))
    out_df = pd.DataFrame(out)
    return out_df


def plot_resids(sdp_df, pep_df, samples_df, K_max=7):
    ista = sdp_df[sdp_df['alg'] == 'ista']
    fista = sdp_df[sdp_df['alg'] == 'fista']

    ista_resids = ista['sdp_objval']
    fista_resids = fista['sdp_objval']

    ista_pep = pep_df['ista_tau']
    fista_pep = pep_df['fista_tau']

    sample_max_df = samples_to_max(samples_df)
    ista_samples = sample_max_df['ista']
    fista_samples = sample_max_df['fista']

    fig, ax = plt.subplots()
    ax.plot(range(1, K_max + 1), ista_resids, marker='<', label='ISTA')
    ax.plot(range(1, K_max + 1), fista_resids, marker='>', label='FISTA')

    ax.plot(range(1, K_max + 1), ista_pep[:K_max], label='ISTA pep')
    ax.plot(range(1, K_max + 1), fista_pep[:K_max], label='FISTA pep')

    ax.plot(range(1, K_max + 1), ista_samples[:K_max], label='ISTA samples')
    ax.plot(range(1, K_max + 1), fista_samples[:K_max], label='FISTA samples')

    ax.set_xlabel('$K$')
    ax.set_ylabel('Worst case fixed-point residual')
    ax.set_yscale('log')

    plt.legend()
    plt.show()
    # plt.savefig('plots/ista_fista.pdf')


def main():
    sdp_df = pd.read_csv('data/ISTA_K1_7_highacc.csv')
    samples_df = pd.read_csv('data/samples.csv')
    pep_df = pd.read_csv('data/pep.csv')

    plot_resids(sdp_df, pep_df, samples_df)


if __name__ == '__main__':
    main()
