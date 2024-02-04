import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def plot_sdp(sdp_df, t_keep, opt=3, outf=None):
    print(sdp_df)
    t_vals = sdp_df['t'].unique()
    print(t_vals)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$K$')
    ax.set_ylabel('Worst case VPSDP fixed-point residual')
    ax.set_yscale('log')
    ax.set_title('Strongly Convex NNLS, Fixed Stepsizes')
    incl_tstar = True

    markers = ['.',',','o','v','^','<','>']
    colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    if outf is not None and 'nonstrong' in outf:
        incl_tstar = False
        markers.pop(3)
        colors.pop(3)
        ax.set_title('Nonstrongly Convex NNLS, Fixed Stepsizes')


    print(t_keep)
    for i, t in enumerate(t_vals):
        if i not in t_keep:
            continue
        df_t = sdp_df[sdp_df['t'] == t]
        K_vals = df_t['K'].unique()
        if incl_tstar and i == opt:
            t_label = f'$t={t:.4f} = t^\star$'
        else:
            t_label = f'$t={t:.4f}$'
        # label = f'$t_{i+1}$'
        # print(t_label)
        ax.plot(K_vals, df_t['sdp_objval'], label=t_label, marker=markers[i], color=colors[i])

    ax.legend()
    ax.set_xticks(K_vals)
    fig.tight_layout()
    # plt.show()
    if outf is None:
        plt.savefig('plots/strong_NNLS_gridt.pdf')
    else:
        plt.savefig(outf)


def samples_to_max(samples_df, K_des=6):
    samples_dfK = samples_df[samples_df['K'] == K_des]
    samples_df['t'].unique()
    max_conv_resid = samples_dfK.groupby(['t']).max()
    return max_conv_resid


def plot_sdp_single_t(sdp_df, samples_df, pep_df, t_keep, K_des=4):
    t_vals = sdp_df['t'].unique()
    sdp_dfK = sdp_df[sdp_df['K'] == K_des]
    samples_df[samples_df['K'] == K_des]
    pep_df[pep_df['K'] == K_des]

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('Worst case fixed-point residual')
    # ax.set_xscale('log')
    ax.set_yscale('log')

    markers = ['o','v','^','<','>']
    ax.plot(np.array(t_vals)[t_keep], np.array(sdp_dfK['sdp_objval'])[t_keep], label='SDP', marker=markers[0])
    # ax.plot(np.array(t_vals)[t_keep], np.array(pep_dfK['tau'])[t_keep], label='PEP', color='green', marker='^')
    # ax.plot(t_vals, )
    # ax.tick_params(axis='x', color='r', labelcolor='r')
    # ax.get_xaxis().set_visible(False)
    plt.title(f'$K={K_des}$')
    plt.axvline(x=t_vals[3], color='black', linestyle='dashed', label='theory optimal')
    plt.axvline(x=t_vals[2], color='black', linestyle='solid', label='best empirical performance')

    max_sample_resid_df = samples_to_max(samples_df, K_des=K_des)

    resids = np.array(max_sample_resid_df['resid'])[t_keep]
    # resids[0] -= .1

    ax.plot(np.array(t_vals)[t_keep], resids,
            label='empirical sample max', marker=markers[1])
    ax.legend()
    # plt.ticklabel_format(style='plain', axis='y')
    # plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
    # plt.gca().yaxis.set_minor_formatter(mticker.ScalarFormatter())
    fig.tight_layout()
    # plt.show()
    plt.savefig(f'plots/K{K_des}_comparison.pdf')


def plot_sdp_single_t_pep(sdp_df, samples_df, pep_df, t_keep, K_des=4):
    t_vals = sdp_df['t'].unique()
    sdp_dfK = sdp_df[sdp_df['K'] == K_des]
    samples_dfK = samples_df[samples_df['K'] == K_des]
    pep_dfK = pep_df[pep_df['K'] == K_des]


    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('Worst case fixed-point residual')
    # ax.set_xscale('log')
    ax.set_yscale('log')

    # markers = ['o','v','^','<','>']
    ax.plot(np.array(t_vals)[t_keep], np.array(sdp_dfK['sdp_objval'])[t_keep], label='VPSDP', color='b', marker='<')
    ax.plot(np.array(t_vals)[t_keep], np.array(pep_dfK['tau'])[t_keep], label='PEP', color='g', marker='o')
    ax.plot(np.array(t_vals)[t_keep], np.array(samples_dfK['resid'])[t_keep], label='Sample Max', color='r', marker='x')
    # ax.plot(t_vals, )
    # ax.tick_params(axis='x', color='r', labelcolor='r')
    # ax.get_xaxis().set_visible(False)
    plt.title(f'Strongly Convex NNLS, $K={K_des}$')
    # plt.axvline(x=t_vals[3], color='black', linestyle='dashed', label='theory optimal')
    plt.axvline(x=t_vals[1], color='black', linestyle='dashed', label='Best PEP bound')
    plt.axvline(x=t_vals[2], color='black', linestyle='solid', label='Best VPSDP/Sample bound')

    # max_sample_resid_df = samples_to_max(samples_df, K_des=K_des)
    # ax.plot(np.array(t_vals)[t_keep], np.array(max_sample_resid_df['resid'])[t_keep],
    #         label='empirical sample max', marker=markers[1])
    # ax.legend()
    # plt.ticklabel_format(style='plain', axis='y')
    # plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
    # plt.gca().yaxis.set_minor_formatter(mticker.ScalarFormatter())

    # ax.legend(loc='center right', bbox_to_anchor=(1, 0.5))
    ax.legend(fontsize='16')
    fig.tight_layout()
    # plt.show()

    plt.savefig(f'plots/K{K_des}_comparison_with_pep.pdf')


def main():
    # sdp_df = pd.read_csv('data/sdp_data.csv')
    # sdp_df = pd.read_csv('data/NNLS_spread_t.csv')
    sdp_df = pd.read_csv('data/NNLS_spreadt_halfc.csv')
    # sample_df = pd.read_csv('data/sample_data.csv')
    samples_max_df = pd.read_csv('data/sample_max.csv')
    pep_df = pd.read_csv('data/pep_data.csv')

    t_keep = np.array([0, 1, 2, 3, 4, 5])
    # plot_sdp(sdp_df, t_keep)
    # plot_sdp_single_t(sdp_df, samples_df, pep_df, t_keep, K_des=7)
    plot_sdp_single_t_pep(sdp_df, samples_max_df, pep_df, t_keep, K_des=4)

    # nonstrong_sdpdf = pd.read_csv('data/nonstrong_grid_sdp.csv')
    # plot_sdp(nonstrong_sdpdf, t_keep, outf='plots/nonstrong_NNLS_gridt.pdf')


if __name__ == '__main__':
    main()
