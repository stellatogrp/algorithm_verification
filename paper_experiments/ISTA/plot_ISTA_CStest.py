import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def plot_sdp(dff, dfi, dfg):
    dff_obj = dff['sdp_objval']
    dfi_obj = dfi['sdp_objval']
    dfg_obj = dfg['glob_objval']

    dff_time = dff['sdp_solvetime']
    dfi_time = dfi['sdp_solvetime']
    dfg_time = dfg['glob_runtime']

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$K$')
    ax.set_ylabel('Worst case fixed-point residual')
    # ax.set_xscale('log')
    ax.set_yscale('log')

    fK = range(1, 8)
    iK = range(1, 6)
    gK = range(1, 6)

    plt.plot(fK, dff_obj, label='full rlt')
    plt.plot(iK, dfi_obj, label='indiv rlt')
    plt.plot(gK, dfg_obj, label='glob after 2 min')

    plt.legend()

    plt.savefig('plots/test_resids.pdf')

    plt.cla()
    plt.clf()

    fig, ax = plt.subplots(1, 1)
    ax.set_xlabel('$K$')
    ax.set_ylabel('time (s)')
    # ax.set_xscale('log')
    ax.set_yscale('log')

    plt.plot(fK, dff_time, label='full rlt')
    plt.plot(iK, dfi_time, label='indiv rlt')
    plt.plot(gK, dfg_time, label='glob after 2 min')

    plt.legend()

    plt.savefig('plots/test_times.pdf')


def main():
    full_df = pd.read_csv('data/test_fullrlt.csv')
    indiv_df = pd.read_csv('data/test_indivrlt.csv')
    glob_df = pd.read_csv('data/glob_K5.csv')

    plot_sdp(full_df, indiv_df, glob_df)
    # plot_sdp_single_t(sdp_df, samples_df, pep_df, t_keep)


if __name__ == '__main__':
    main()
