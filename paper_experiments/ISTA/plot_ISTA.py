import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def plot_resids(sdp_df, K_max=7):
    ista = sdp_df[sdp_df['alg'] == 'ista']
    fista = sdp_df[sdp_df['alg'] == 'fista']

    ista_resids = ista['sdp_objval']
    fista_resids = fista['sdp_objval']

    fig, ax = plt.subplots()
    ax.plot(range(1, K_max + 1), ista_resids, marker='<', label='ISTA')
    ax.plot(range(1, K_max + 1), fista_resids, marker='>', label='FISTA')

    ax.set_xlabel('$K$')
    ax.set_ylabel('Worst case fixed-point residual')
    ax.set_yscale('log')

    plt.legend()
    # plt.show()
    plt.savefig('plots/ista_fista.pdf')


def main():
    sdp_df = pd.read_csv('data/ISTA_K1_7_highacc.csv')
    # samples_df = pd.read_csv('data/num_sample.csv')
    # pep_df = pd.read_csv('data/num_pep.csv')

    plot_resids(sdp_df)


if __name__ == '__main__':
    main()
