import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def plot_resids(sdp_df, samples_df, pep_df):
    # print(sdp_df)
    # t_vals = sdp_df['t'].unique()
    # print(t_vals)

    # fig, ax = plt.subplots(1, 1)
    # ax.set_xlabel('$K$')
    # ax.set_ylabel('Max SDP obejctive')
    # ax.set_yscale('log')

    # markers = ['.',',','o','v','^','<','>']

    # for i, t in enumerate(t_vals):
    #     df_t = sdp_df[sdp_df['t'] == t]
    #     K_vals = df_t['K'].unique()
    #     ax.plot(K_vals, df_t['sdp_objval'], label=f'$t_{i+1}$', marker=markers[i])

    # ax.legend()
    # fig.tight_layout()
    # # plt.show()
    # plt.savefig('plots/sdp_obj_all_K.pdf')
    samp_cs_resids, samp_ws_resids, samp_heur_resids = samples_to_max(samples_df)

    # print(samp_cs_resids, samp_ws_resids)
    pep_cs_tau, pep_ws_tau, pep_heur_tau = pep_to_res(pep_df)

    sdp_cs, sdp_ws, sdp_heur = sdp_to_res(sdp_df)

    K_vals = sdp_df['K'].unique()

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(9, 6), sharey=True)
    ax0.set_xlabel('$K$')
    ax1.set_xlabel('$K$')
    ax2.set_xlabel('$K$')
    ax0.set_ylabel('Worst case fixed-point residual')
    ax0.set_yscale('log')

    sdp_m = '<'
    pep_m = 'o'
    samp_m = 'x'

    sdp_color = 'b'
    pep_color = 'g'
    samp_color = 'r'

    # ax0.plot(K_vals, sdp_cs, marker=sdp_m, color=sdp_color, label='SDP, cold')
    # ax0.plot(K_vals, pep_cs_tau, marker=pep_m, color=pep_color, label='PEP, cold')
    # ax0.plot(K_vals, samp_cs_resids, marker=samp_m, color=samp_color, label='Samples, cold')
    # ax0.set_title('Cold start')
    ax0.plot(K_vals, sdp_cs, marker=sdp_m, color=sdp_color)
    ax0.plot(K_vals, pep_cs_tau, marker=pep_m, color=pep_color)
    ax0.plot(K_vals, samp_cs_resids, marker=samp_m, color=samp_color)
    ax0.set_xticks(K_vals)
    ax0.set_title('Cold Start')

    ax1.plot(K_vals, sdp_ws, marker=sdp_m, color=sdp_color, label='VPSDP')
    ax1.plot(K_vals, pep_ws_tau, marker=pep_m, color=pep_color, label='PEP')
    ax1.plot(K_vals, samp_ws_resids, marker=samp_m, color=samp_color, label='SM')
    ax1.set_xticks(K_vals)
    ax1.set_title('Warm Start')

    ax2.set_title('Heuristic Start')
    # ax2.plot(K_vals, sdp_heur, marker=sdp_m, color=sdp_color, label='SDP, heuristic')
    # ax2.plot(K_vals, pep_heur_tau, marker=pep_m, color=pep_color, label='PEP, heuristic')
    # ax2.plot(K_vals, samp_heur_resids, marker=samp_m, color=samp_color, label='Samples, heuristic')

    ax2.plot(K_vals, sdp_heur, marker=sdp_m, color=sdp_color)
    ax2.plot(K_vals, pep_heur_tau, marker=pep_m, color=pep_color)
    ax2.plot(K_vals, samp_heur_resids, marker=samp_m, color=samp_color)
    ax2.set_xticks(K_vals)

    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.set_ylim([None, 10])
    # ax0.legend(loc='upper right')
    # ax1.legend(loc='upper right')
    # plt.legend()

    fig.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.11))
    # plt.suptitle(r'NUM example, $R \in { \bf R}^{10\times 5}$')
    plt.suptitle(r'Network utility maximization')
    plt.tight_layout()

    axes = [ax0, ax1, ax2]
    for ax in axes:
        print(ax.get_position())
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0+.05, pos.width, pos.height]
        ax.set_position(new_pos)

    print(samp_cs_resids, samp_ws_resids, samp_heur_resids)
    # plt.show()
    plt.savefig('plots/NUM.pdf')


def samples_to_max(samples_df):
    # print(samples_df)
    df_cs = samples_df[samples_df['type'] == 'cs']
    df_ws = samples_df[samples_df['type'] == 'ws']
    df_heur = samples_df[samples_df['type'] == 'heur']
    cs_resids = df_cs.groupby(['K'])
    ws_resids = df_ws.groupby(['K'])
    heur_resids = df_heur.groupby(['K'])
    # print(cs_resids['res'].max())

    return cs_resids['res'].max(), ws_resids['res'].max(), heur_resids['res'].max()


def pep_to_res(pep_df):
    df_cs = pep_df[pep_df['type'] == 'cs']
    df_ws = pep_df[pep_df['type'] == 'ws']
    df_heur = pep_df[pep_df['type'] == 'heur']

    return df_cs['tau'], df_ws['tau'], df_heur['tau']


def sdp_to_res(sdp_df):
    # df_cs = sdp_df[sdp_df["warm_start"] is False]
    # df_ws = sdp_df[sdp_df["warm_start"] is True]
    df_cs = sdp_df[sdp_df['init_type'] == 'cs']
    df_ws = sdp_df[sdp_df['init_type'] == 'ws']
    df_heur = sdp_df[sdp_df['init_type'] == 'heur']

    return df_cs['sdp_objval'], df_ws['sdp_objval'], df_heur['sdp_objval']


def main():
    # sdp_df = pd.read_csv('data/NUM_K1_5_highacc.csv')
    # samples_df = pd.read_csv('data/num_sample.csv')
    # pep_df = pd.read_csv('data/num_pep.csv')

    sdp_df = pd.read_csv('data/NUM_seed0_rad04.csv')
    samples_df = pd.read_csv('data/new_num_sample.csv')
    pep_df = pd.read_csv('data/new_num_pep.csv')

    plot_resids(sdp_df, samples_df, pep_df)


if __name__ == '__main__':
    main()
