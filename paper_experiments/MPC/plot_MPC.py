import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def plot_resids(sdp_df, K_max=6):
    rhoconst = sdp_df[sdp_df['rho'] == 'rho_const']
    rhoadj = sdp_df[sdp_df['rho'] == 'rho_adj']

    rhoconst_resids = rhoconst['sdp_objval']
    rhoadj_resids = rhoadj['sdp_objval']

    fig, ax = plt.subplots()
    ax.plot(range(1, K_max + 1), rhoconst_resids, marker='<', label='rho const')
    ax.plot(range(1, K_max + 1), rhoadj_resids, marker='>', label='rho nonconst')

    ax.set_xlabel('$K$')
    ax.set_ylabel('Worst case fixed-point residual')
    ax.set_yscale('log')

    ax.set_ylim(bottom=1e-1)

    plt.legend()
    plt.show()
    # plt.savefig('plots/mpc.pdf')


def main():
    sdp_df = pd.read_csv('data/MPC.csv')
    # samples_df = pd.read_csv('data/samples.csv')
    # pep_df = pd.read_csv('data/pep.csv')

    plot_resids(sdp_df)


if __name__ == '__main__':
    main()
