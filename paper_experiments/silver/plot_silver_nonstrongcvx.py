import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def plot_resids(sdp_df, pep_df, sample_df, fixed_sdp, fixed_pep, fixed_samp, K_max=7):
    silver_sdp = sdp_df[sdp_df['sched'] == 'silver']['sdp_objval']
    fixed_sdp = sdp_df[sdp_df['sched'] == 'fixed']['sdp_objval']

    silver_pep = pep_df[pep_df['sched'] == 'silver']['tau']
    fixed_pep = pep_df[pep_df['sched'] == 'fixed']['tau']

    silver_samp = sample_df[sample_df['sched'] == 'silver']['resid']
    fixed_samp = sample_df[sample_df['sched'] == 'fixed']['resid']

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
    K_vals = range(1, K_max + 1)

    ax0.plot(K_vals, silver_sdp[:K_max], marker=sdp_m, color=sdp_color, label='VPSDP')
    ax0.plot(K_vals, silver_pep[:K_max], marker=pep_m, color=pep_color, label='PEP')
    ax0.plot(K_vals, silver_samp[:K_max], marker=samp_m, color=samp_color, label='SM')
    ax0.set_xticks(K_vals)
    ax0.set_title('Silver Schedule')

    ax1.plot(K_vals, fixed_sdp[:K_max], marker=sdp_m, color=sdp_color)
    ax1.plot(K_vals, fixed_pep[:K_max], marker=pep_m, color=pep_color)
    ax1.plot(K_vals, fixed_samp[:K_max], marker=samp_m, color=samp_color)
    ax1.set_xticks(K_vals)
    ax1.set_title(r'Fixed $t=1.5/L$')

    fig.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.11))

    plt.suptitle(r'NNLS, Nonstrongly Convex Instance')
    plt.tight_layout()

    axes = [ax0, ax1]
    for ax in axes:
        print(ax.get_position())
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0+.05, pos.width, pos.height]
        ax.set_position(new_pos)

    # plt.show()
    plt.savefig('plots/silver_nonstrongcvx.pdf')


def main():
    sdp_df = pd.read_csv('data/NNLS_nonstrong_silver.csv')
    samples_df = pd.read_csv('data/sample_max.csv')
    pep_df = pd.read_csv('data/nonstrong_pep_data_quad.csv')

    fixedt_sdp_df = pd.read_csv('../NNLS/data/nonstrong_grid_sdp.csv')
    fixedt_samples_df = pd.read_csv('../NNLS/data/nonstrong_grid_sample_max.csv')
    fixedt_pep_df = pd.read_csv('../NNLS/data/nonstrong_grid_pep_data_quad.csv')

    fixedt_sdp_df = fixedt_sdp_df[fixedt_sdp_df['t'] == 0.015]
    fixedt_samples_df = fixedt_samples_df[fixedt_samples_df['t'] == 0.015]
    fixedt_pep_df = fixedt_pep_df[fixedt_pep_df['t'] == 0.015]

    plot_resids(sdp_df, pep_df, samples_df, fixedt_sdp_df, fixedt_pep_df, fixedt_samples_df)


if __name__ == '__main__':
    main()
