import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def plot_resids(sdp_df, samp_pep_df, K_max=6, single_plot=False):
    rhoconst = sdp_df[sdp_df['rho'] == 'rho_const']
    rhoadj = sdp_df[sdp_df['rho'] == 'rho_adj']

    rhoconst_resids = rhoconst['sdp_objval']
    rhoadj_resids = rhoadj['sdp_objval']

    samp_const_resids = samp_pep_df['const_samp_max']
    samp_adj_resids = samp_pep_df['adj_samp_max']
    const_pep = samp_pep_df['const_pep']

    K_vals = range(1, K_max + 1)

    if single_plot:
        fig, ax = plt.subplots()
        ax.plot(range(1, K_max + 1), rhoconst_resids[:K_max], marker='<', label=r'Single $\rho$')
        ax.plot(range(1, K_max + 1), rhoadj_resids[:K_max], marker='>', label=r'Adj. $\rho$')
        ax.plot(range(1, K_max + 1), samp_const_resids[:K_max], marker='x', label=r'Sampled Single $\rho$')
        ax.plot(range(1, K_max + 1), samp_adj_resids[:K_max], marker='o', label=r'Sampled Adj. $\rho$')
        ax.plot(range(1, K_max + 1), const_pep[:K_max], marker='s', label=r'PEP, Single $\rho$')

        ax.set_xlabel('$K$')
        ax.set_ylabel('Worst case fixed-point residual')
        ax.set_yscale('log')

        plt.legend()

    # ax.set_ylim(bottom=1e-0)
    else:
        sdp_m = '<'
        pep_m = 'o'
        samp_m = 'x'

        sdp_color = 'b'
        pep_color = 'g'
        samp_color = 'r'

        fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
        ax0.set_xlabel('$K$')
        ax1.set_xlabel('$K$')
        ax0.set_ylabel('Worst case fixed-point residual')
        ax0.set_yscale('log')

        ax0.plot(K_vals, rhoconst_resids[:K_max], marker=sdp_m, color=sdp_color, label='VPSDP')
        ax0.plot(K_vals, const_pep[:K_max], marker=pep_m, color=pep_color, label='PEP')
        ax0.plot(K_vals, samp_const_resids[:K_max], marker=samp_m, color=samp_color, label='Sample Max')
        ax0.set_xticks(K_vals)
        ax0.set_title(r'Scalar $\rho$')

        ax1.plot(K_vals, rhoadj_resids[:K_max], marker=sdp_m, color=sdp_color)
        # ax1.plot(K_vals, fista_pep[:K_max], marker=pep_m, color=pep_color)
        ax1.plot(K_vals, samp_adj_resids[:K_max], marker=samp_m, color=samp_color)
        ax1.set_title(r'Diagonal $\rho$')
        ax1.set_xticks(K_vals)

        fig.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.11))
        plt.suptitle(r'Model predictive control')
        plt.tight_layout()

        axes = [ax0, ax1]
        for ax in axes:
            print(ax.get_position())
            pos = ax.get_position()
            new_pos = [pos.x0, pos.y0+.05, pos.width, pos.height]
            ax.set_position(new_pos)

    # plt.show()
    plt.savefig('plots/mpc.pdf')


def main():
    sdp_df = pd.read_csv('data/MPC_ws_1e-3.csv')
    # samples_df = pd.read_csv('data/samples.csv')
    # pep_df = pd.read_csv('data/pep.csv')
    samp_pep_df = pd.read_csv('data/ws_samp_pep.csv')

    plot_resids(sdp_df, samp_pep_df)


if __name__ == '__main__':
    main()
