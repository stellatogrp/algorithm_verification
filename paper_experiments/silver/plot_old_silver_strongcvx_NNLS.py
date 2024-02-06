import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def plot_stepsizes(df_silver, df_othert, max_plot=None):
    K_vals = df_silver['K'].unique()

    if max_plot is None:
        max_plot = K_vals.shape[0]

    df_sonly = df_silver[df_silver['sched'] == 'silver']
    df_topt = df_silver[df_silver['sched'] == 't_opt']
    t_opt = df_topt['t'].unique()[0]
    other_tvals = df_othert['t'].unique()

    silver_steps = df_sonly['t']
    fig, ax = plt.subplots()
    ax.plot(K_vals[:max_plot], silver_steps[:max_plot], marker='o', label='Silver', color='black')

    colors = ['blue', 'red', 'orange']
    for i, t in enumerate(other_tvals):
        # ax.plot(K_vals[:max_plot], df_othert[df_othert['t'] == t]['sdp_objval'][:max_plot], label=f'$t = {t:.4f}$', color=colors[i], marker=markers[i])
        ax.axhline(t, label=f'$t = {t:.4f}$', color=colors[i])

    ax.axhline(t_opt, label=f'$t^\star = {t_opt:.4f}$', color='green')

    plt.legend()
    ax.set_title('Step sizes')
    ax.set_xlabel('$K$')
    ax.set_ylim(ymin=0)
    # plt.show()
    plt.savefig('plots/strongcvx/silversteps_m15n8.pdf')


def plot_sdp_objs(df_silver, df_othert, max_plot=None):
    K_vals = df_silver['K'].unique()

    if max_plot is None:
        max_plot = K_vals.shape[0]

    df_sonly = df_silver[df_silver['sched'] == 'silver']
    df_topt = df_silver[df_silver['sched'] == 't_opt']
    t_opt = df_topt['t'].unique()[0]
    other_tvals = df_othert['t'].unique()

    fig, ax = plt.subplots()

    ax.set_yscale('log')
    ax.set_xlabel('$K$')
    ax.set_ylabel('Worst case fixed point residual')
    # ax.set_title(r'$A \in {\bf R}^{m \times n}, m=15, n=8, c_{b(\theta)} = 10({\bf 1}), r_{b(\theta)}=1/2$')
    ax.set_title(r'$A \in {\bf R}^{m \times n}, m=15, n=8$')

    ax.plot(K_vals[:max_plot], df_sonly['sdp_objval'][:max_plot], color='black', label='Silver', marker='o')

    markers = ['>', '<', 'v']
    colors = ['blue', 'red', 'orange']
    for i, t in enumerate(other_tvals):
        ax.plot(K_vals[:max_plot], df_othert[df_othert['t'] == t]['sdp_objval'][:max_plot], label=f'$t = {t:.4f}$', color=colors[i], marker=markers[i])

    ax.plot(K_vals[:max_plot], df_topt['sdp_objval'][:max_plot], marker='^', label=f'$t^\star = {t_opt:.4f}$', color='green')

    ax.legend()
    # plt.show()
    plt.savefig('plots/strongcvx/sdpobjs_m15n8.pdf')


def samples_to_max(samples_df, K_des=6, key='t'):
    samples_dfK = samples_df[samples_df['K'] == K_des]
    # samples_df['t'].unique()
    max_conv_resid = samples_dfK.groupby([key]).max()
    return max_conv_resid


# def plot_sdp_single_t(sdp_df, samples_df, pep_df, t_keep, K_des=6):
#     t_vals = sdp_df['t'].unique()
#     sdp_dfK = sdp_df[sdp_df['K'] == K_des]
#     samples_df[samples_df['K'] == K_des]
#     pep_dfK = pep_df[pep_df['K'] == K_des]

#     fig, ax = plt.subplots(1, 1)
#     ax.set_xlabel('$t$')
#     ax.set_ylabel('Worst case fixed-point residual')
#     # ax.set_xscale('log')
#     ax.set_yscale('log')

#     markers = ['o','v','^','<','>']
#     ax.plot(np.array(t_vals)[t_keep], np.array(sdp_dfK['sdp_objval'])[t_keep], label='SDP', marker=markers[0])
#     ax.plot(np.array(t_vals)[t_keep], np.array(pep_dfK['tau'])[t_keep], label='PEP', color='green', marker='^')
#     # ax.plot(t_vals, )
#     # ax.tick_params(axis='x', color='r', labelcolor='r')
#     # ax.get_xaxis().set_visible(False)
#     plt.title(f'$K={K_des}$')
#     plt.axvline(x=t_vals[3], color='black', linestyle='dashed', label='theory optimal')
#     plt.axvline(x=t_vals[2], color='black', linestyle='solid', label='best empirical performance')

#     samples_to_max(samples_df, K_des=K_des)
#     # ax.plot(np.array(t_vals)[t_keep], np.array(max_sample_resid_df['resid'])[t_keep],
#     #         label='empirical sample max', marker=markers[1])
#     ax.legend()
#     # plt.ticklabel_format(style='plain', axis='y')
#     # plt.gca().yaxis.set_major_formatter(mticker.ScalarFormatter())
#     # plt.gca().yaxis.set_minor_formatter(mticker.ScalarFormatter())
#     fig.tight_layout()
#     plt.show()
#     # plt.savefig('plots/K6_comparison.pdf')

#     # plt.savefig('plots/K6_comparison_with_pep.pdf')


def plot_pep_fixed_K(df_s, df_t, samp_dfs, samp_dft, pep_dfs, pep_dft, max_plot=None, K_des=6):
    df_sK = df_s[df_s['K'] == K_des]
    df_tK = df_t[df_t['K'] == K_des]
    # samp_dfsK = samp_dfs[samp_dfs['K'] == K_des]
    # samp_dftK = samp_dft[samp_dft['K'] == K_des]
    samp_dfsK = samples_to_max(samp_dfs, K_des=K_des, key='sched')
    samp_dftK = samples_to_max(samp_dft, K_des=K_des, key='t')
    pep_dfsK = pep_dfs[pep_dfs['K'] == K_des]
    pep_dftK = pep_dft[pep_dft['K'] == K_des]
    t_opt = df_s[df_s['sched'] == 't_opt']['t'].unique()[0]

    t_vals = list(df_tK['t'].unique())
    t_vals.insert(2, t_opt)

    sdpobjs_t = list(df_tK['sdp_objval'])
    # print(df_s[df_s['sched'] == 't_opt'])
    sdpobjs_t.insert(2, df_sK[df_sK['sched'] == 't_opt']['sdp_objval'].unique()[0])
    print(sdpobjs_t)

    pep_t = list(pep_dftK['tau'])
    pep_t.insert(2, pep_dfsK[pep_dfsK['sched'] == 't_opt']['tau'].unique()[0])
    # print(pep_t)

    print(samp_dfsK, samp_dftK)
    samp_t = list(samp_dftK['resid'])
    samp_t.insert(2, samp_dfsK.at['t_opt', 'resid'])

    fig, ax = plt.subplots()
    ax.set_xlabel('$t$')
    ax.set_ylabel('Worst case fixed-point residual')
    ax.set_yscale('log')

    ax.plot(t_vals, sdpobjs_t, label='SDP', color='blue', marker='o')
    ax.plot(t_vals, pep_t, label='PEP', color='green', marker='^')
    plt.axvline(t_vals[2], label='Theory optimal', color='red', linestyle='dashed')
    plt.axvline(t_vals[1], label='Best empirical performance', color='red')
    plt.title(f'$K = {K_des}$')

    # print(df_sK[df_sK['sched'] == 'silver']['sdp_objval'].unique()[0])
    # exit(0)
    ax.axhline(df_sK[df_sK['sched'] == 'silver']['sdp_objval'].unique()[0], label='Silver', color='black')

    ax.legend()

    plt.savefig('plots/strongcvx/K6withPEP_m15n8.pdf')
    plt.show()

    fig, ax = plt.subplots()
    ax.set_xlabel('$t$')
    ax.set_ylabel('Worst case fixed-point residual')
    ax.set_yscale('log')

    ax.plot(t_vals, sdpobjs_t, label='SDP', color='blue', marker='o')
    ax.plot(t_vals, samp_t, label='empirical sample max', color='orange', marker='v')

    plt.axvline(t_vals[2], label='Theory optimal', color='red', linestyle='dashed')
    plt.axvline(t_vals[1], label='Best empirical performance', color='red')
    ax.axhline(df_sK[df_sK['sched'] == 'silver']['sdp_objval'].unique()[0], label='Silver', color='black')

    ax.set_ylim(ymax=1)
    plt.title(f'$K = {K_des}$')

    ax.legend()
    plt.savefig('plots/strongcvx/K6withsamples_m15n8.pdf')
    plt.show()


def main():
    sdp_silver_df = pd.read_csv('data/strongcvx/sdp_silver_m15n8_rhalf.csv')
    sdp_tfixed_df = pd.read_csv('data/strongcvx/sdp_tfixed_m15n8_rhalf.csv')

    sample_silver_df = pd.read_csv('data/strongcvx/sample_silver_m15n8.csv')
    sample_tfixed_df = pd.read_csv('data/strongcvx/sample_tfixed_m15n8.csv')

    pep_silver_df = pd.read_csv('data/strongcvx/pep_silver_m15n8.csv')
    pep_tfixed_df = pd.read_csv('data/strongcvx/pep_tfixed_m15n8.csv')

    max_plot = 7
    K_des = 6
    plot_sdp_objs(sdp_silver_df, sdp_tfixed_df, max_plot=max_plot)
    plot_stepsizes(sdp_silver_df, sdp_tfixed_df, max_plot=max_plot)
    plot_pep_fixed_K(sdp_silver_df, sdp_tfixed_df,
                     sample_silver_df, sample_tfixed_df,
                     pep_silver_df, pep_tfixed_df,
                     max_plot=max_plot, K_des=K_des)


if __name__ == '__main__':
    main()
