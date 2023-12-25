import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
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
    samp_cs_resids, samp_ws_resids = samples_to_max(samples_df)
    # print(samp_cs_resids, samp_ws_resids)
    pep_cs_tau, pep_ws_tau = pep_to_res(pep_df)

    sdp_cs, sdp_ws = sdp_to_res(sdp_df)

    K_vals = sdp_df['K'].unique()

    fig, (ax0, ax1) = plt.subplots(1, 2, sharey=True)
    ax0.set_xlabel('$K$')
    ax1.set_xlabel('$K$')
    ax0.set_ylabel('Worst case fixed-point residual')
    ax0.set_yscale('log')

    sdp_m = '<'
    pep_m = 'v'
    samp_m = 'o'

    ax0.plot(K_vals, sdp_cs, marker=sdp_m, linestyle='dashed', label='SDP, cold')
    ax0.plot(K_vals, pep_cs_tau, marker=pep_m, linestyle='dashed', label='PEP, cold')
    ax0.plot(K_vals, samp_cs_resids, marker=samp_m, linestyle='dashed', label='Samples, cold')

    ax1.plot(K_vals, sdp_ws, marker=sdp_m, label='SDP, warm')
    ax1.plot(K_vals, pep_ws_tau, marker=pep_m, label='PEP, warm')
    ax1.plot(K_vals, samp_ws_resids, marker=samp_m, label='Samples, warm')

    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    # ax.set_ylim([None, 10])
    ax0.legend(loc='upper right')
    ax1.legend(loc='upper right')
    plt.suptitle(r'NUM example, $R \in { \bf R}^{10\times 5}$')
    plt.tight_layout()
    # plt.show()
    plt.savefig('plots/NUM.pdf')


def samples_to_max(samples_df):
    # print(samples_df)
    df_cs = samples_df[samples_df['type'] == 'cs']
    df_ws = samples_df[samples_df['type'] == 'ws']
    cs_resids = df_cs.groupby(['K'])
    ws_resids = df_ws.groupby(['K'])
    # print(cs_resids['res'].max())

    return cs_resids['res'].max(), ws_resids['res'].max()


def pep_to_res(pep_df):
    df_cs = pep_df[pep_df['type'] == 'cs']
    df_ws = pep_df[pep_df['type'] == 'ws']

    return df_cs['tau'], df_ws['tau']


def sdp_to_res(sdp_df):
    # df_cs = sdp_df[sdp_df["warm_start"] is False]
    # df_ws = sdp_df[sdp_df["warm_start"] is True]
    df_cs = sdp_df[sdp_df['init_type'] == 'cs']
    df_ws = sdp_df[sdp_df['init_type'] == 'ws']

    return df_cs['sdp_objval'], df_ws['sdp_objval']


def main():
    sdp_df = pd.read_csv('data/NUM_K1_5_highacc.csv')
    samples_df = pd.read_csv('data/num_sample.csv')
    pep_df = pd.read_csv('data/num_pep.csv')

    plot_resids(sdp_df, samples_df, pep_df)


if __name__ == '__main__':
    main()
