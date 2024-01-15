import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def plot_resids(sdp_df, samp_pep_df, K_max=7):
    rhoconst = sdp_df[sdp_df['rho'] == 'rho_const']
    rhoadj = sdp_df[sdp_df['rho'] == 'rho_adj']

    rhoconst_resids = rhoconst['sdp_objval']
    rhoadj_resids = rhoadj['sdp_objval']

    samp_const_resids = samp_pep_df['const_samp_max']
    samp_adj_resids = samp_pep_df['adj_samp_max']
    const_pep = samp_pep_df['const_pep']

    fig, ax = plt.subplots()
    ax.plot(range(1, K_max + 1), rhoconst_resids[:K_max], marker='<', label=r'Single $\rho$')
    ax.plot(range(1, K_max + 1), rhoadj_resids[:K_max], marker='>', label=r'Adj. $\rho$')
    ax.plot(range(1, K_max + 1), samp_const_resids[:K_max], marker='x', label=r'Sampled Single $\rho$')
    ax.plot(range(1, K_max + 1), samp_adj_resids[:K_max], marker='o', label=r'Sampled Adj. $\rho$')
    ax.plot(range(1, K_max + 1), const_pep[:K_max], marker='s', label=r'PEP, Single $\rho$')

    ax.set_xlabel('$K$')
    ax.set_ylabel('Worst case fixed-point residual')
    ax.set_yscale('log')

    # ax.set_ylim(bottom=1e-0)

    plt.legend()
    plt.show()
    # plt.savefig('plots/mpc.pdf')


def main():
    sdp_df = pd.read_csv('data/MPC_ws.csv')
    # samples_df = pd.read_csv('data/samples.csv')
    # pep_df = pd.read_csv('data/pep.csv')
    samp_pep_df = pd.read_csv('data/ws_samp_pep.csv')

    plot_resids(sdp_df, samp_pep_df)


if __name__ == '__main__':
    main()
