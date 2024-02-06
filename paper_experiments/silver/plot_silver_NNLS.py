import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def get_t_opt(tval_set):
    return next(iter(tval_set - set(['silver'])))


def get_silver_steps(K):
    rho = 1 + np.sqrt(2)
    return np.array([1 + rho ** ((k & -k).bit_length()-2) for k in range(1, K+1)])


def plot_sdp(sdp_df):
    K_vals = sdp_df['K'].unique()
    t_vals = sdp_df['t'].unique()
    t_opt = get_t_opt(set(t_vals))
    df_topt = sdp_df[sdp_df['t'] == t_opt]
    df_silver = sdp_df[sdp_df['t'] == 'silver']

    t_opt = float(t_opt)
    print(df_topt)
    print(df_silver)

    L = sdp_df['L'].unique()[0]
    silver_steps = get_silver_steps(int(np.max(K_vals))) / L
    print(silver_steps)
    print(L)

    silver_objs = df_silver['sdp_objval']
    topt_objs = df_topt['sdp_objval']

    fig, ax = plt.subplots()
    ax.plot(K_vals, silver_objs, label='silver', marker='>')

    ax.plot(K_vals, topt_objs, label=f'$t^\star = {t_opt:.3f}$', marker='^')

    ax.set_yscale('log')
    ax.set_xlabel('$K$')
    ax.set_ylabel('Worst case fixed point residual')
    ax.set_title('NNLS SDP Relaxation')

    plt.legend()

    # plt.show()
    plt.savefig('plots/sdpobjs.pdf')

def plot_stepsizes(sdp_df):
    K_vals = sdp_df['K'].unique()
    t_vals = sdp_df['t'].unique()
    t_opt = get_t_opt(set(t_vals))
    t_opt = float(t_opt)

    L = sdp_df['L'].unique()[0]
    silver_steps = get_silver_steps(int(np.max(K_vals))) / L
    print(silver_steps)
    print(L)

    fig, ax = plt.subplots()
    ax.scatter(K_vals, silver_steps, label='silver', color='black')
    ax.axhline(2/L, label=r'$2/L$', color='black')
    ax.axhline(t_opt, label=f'$t^\star = {t_opt:.3f}$', color='black', linestyle='dashed')
    plt.legend()
    ax.set_title('Step sizes')
    ax.set_xlabel('$K$')
    ax.set_ylim(ymin=0)
    # plt.show()
    plt.savefig('plots/stepsizes.pdf')


def main():
    sdp_df = pd.read_csv('data/silver_NNLS_sdp.csv')
    # glob_df = pd.read_csv('data/silver_NNLS_glob.csv')
    # plot_sdp_single_t(sdp_df, samples_df, pep_df, t_keep)

    plot_sdp(sdp_df)
    # plot_stepsizes(sdp_df)


if __name__ == '__main__':
    main()
