import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def plot_sdp(sdp_df):
    print(sdp_df)
    t_vals = sdp_df['t'].unique()
    print(t_vals)

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$K$')
    ax.set_ylabel('Max SDP obejctive')
    ax.set_yscale('log')

    markers = ['.',',','o','v','^','<','>']

    for i, t in enumerate(t_vals):
        df_t = sdp_df[sdp_df['t'] == t]
        K_vals = df_t['K'].unique()
        ax.plot(K_vals, df_t['sdp_objval'], label=f'$t_{i+1}$', marker=markers[i])

    ax.legend()
    fig.tight_layout()
    # plt.show()
    plt.savefig('plots/sdp_obj_all_K.pdf')


def samples_to_max(samples_df, K_des=6):
    samples_dfK = samples_df[samples_df['K'] == K_des]
    samples_df['t'].unique()
    max_conv_resid = samples_dfK.groupby(['t']).max()
    return max_conv_resid


def plot_sdp_single_t(sdp_df, samples_df, pep_df, K_des=10):
    t_vals = sdp_df['t'].unique()
    sdp_dfK = sdp_df[sdp_df['K'] == K_des]
    samples_df[samples_df['K'] == K_des]
    pep_df[pep_df['K'] == K_des]

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$t$')
    ax.set_ylabel('Maximum fixed point residual')
    # ax.set_xscale('log')
    ax.set_yscale('log')

    markers = ['o','v','^','<','>']
    ax.plot(t_vals, sdp_dfK['sdp_objval'], label='SDP', marker=markers[0])
    # ax.plot(t_vals, pep_dfK['tau'], label='PEP with sampled radius')
    # ax.plot(t_vals, )
    # ax.tick_params(axis='x', color='r', labelcolor='r')
    # ax.get_xaxis().set_visible(False)
    plt.title(f'$K={K_des}$')
    plt.axvline(x=t_vals[4], color='black', linestyle='dashed', label='theory optimal')
    plt.axvline(x=t_vals[2], color='black', linestyle='solid', label='best empirical performance')

    max_sample_resid_df = samples_to_max(samples_df, K_des=K_des)
    ax.plot(t_vals, max_sample_resid_df['resid'], label='empirical sample max', marker=markers[1])
    ax.legend()
    fig.tight_layout()
    plt.show()
    # plt.savefig('plots/K6_comparison.pdf')


def main():
    sdp_df = pd.read_csv('data/sdp_data.csv')
    samples_df = pd.read_csv('data/sample_data.csv')
    pep_df = pd.read_csv('data/pep_data.csv')

    # plot_sdp(sdp_df)
    plot_sdp_single_t(sdp_df, samples_df, pep_df)


if __name__ == '__main__':
    main()
