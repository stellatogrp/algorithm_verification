import matplotlib.pyplot as plt
import pandas as pd

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def samples_to_max(samples_df, K_max=7):
    # print(samples_df)

    out = []
    for i in range(1, K_max + 1):
        ista_max = samples_df[(samples_df['K'] == i)]['ista'].max()
        fista_max = samples_df[(samples_df['K'] == i)]['fista'].max()

        out.append(pd.Series(
            {
                'K': i,
                'ista': ista_max,
                'fista': fista_max,
            }
        ))
    out_df = pd.DataFrame(out)
    return out_df


def plot_resids(sdp_df, pep_df, samples_df, K_max=7, single_plot=False):
    ista = sdp_df[sdp_df['alg'] == 'ista']
    fista = sdp_df[sdp_df['alg'] == 'fista']

    ista_resids = ista['sdp_objval']
    fista_resids = fista['sdp_objval']

    ista_pep = pep_df['ista_tau']
    fista_pep = pep_df['fista_tau']

    sample_max_df = samples_to_max(samples_df)
    ista_samples = sample_max_df['ista']
    fista_samples = sample_max_df['fista']
    print(ista_samples)
    print(fista_samples)

    K_vals = range(1, K_max + 1)

    if single_plot:
        fig, ax = plt.subplots(figsize = (10, 6))
        ax.plot(K_vals, ista_resids, marker='<', label='ISTA')
        ax.plot(K_vals, fista_resids, marker='>', label='FISTA')

        ax.plot(K_vals, ista_pep[:K_max], marker='^', label='ISTA PEP')
        ax.plot(K_vals, fista_pep[:K_max], marker='^', label='FISTA PEP')

        ax.plot(K_vals, ista_samples[:K_max], marker='x', label='Sampled ISTA')
        ax.plot(K_vals, fista_samples[:K_max], marker='x', label='Sampled FISTA')

        ax.set_xlabel('$K$')
        ax.set_ylabel('Worst case fixed-point residual')
        ax.set_yscale('log')

        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()

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

        ax0.plot(K_vals, ista_resids[:K_max], marker=sdp_m, color=sdp_color, label='VPSDP')
        ax0.plot(K_vals, ista_pep[:K_max], marker=pep_m, color=pep_color, label='PEP')
        ax0.plot(K_vals, ista_samples[:K_max], marker=samp_m, color=samp_color, label='SM')
        ax0.set_xticks(K_vals)
        ax0.set_title('ISTA')

        ax1.plot(K_vals, fista_resids[:K_max], marker=sdp_m, color=sdp_color)
        ax1.plot(K_vals, fista_pep[:K_max], marker=pep_m, color=pep_color)
        ax1.plot(K_vals, fista_samples[:K_max], marker=samp_m, color=samp_color)
        ax1.set_xticks(K_vals)
        ax1.set_title('FISTA')

        fig.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.11))

        plt.suptitle(r'Lasso')
        plt.tight_layout()

        axes = [ax0, ax1]
        for ax in axes:
            print(ax.get_position())
            pos = ax.get_position()
            new_pos = [pos.x0, pos.y0+.05, pos.width, pos.height]
            ax.set_position(new_pos)

    # plt.show()
    plt.savefig('plots/ista_fista.pdf')


def plot_global_ista(samples_df):
    glob_df = pd.read_csv('data/ISTA_glob_test.csv')
    print(glob_df)

    ista = glob_df[glob_df['alg'] == 'ista']
    fista = glob_df[glob_df['alg'] == 'fista']

    ista_objs = ista['glob_objval']
    fista_objs = fista['glob_objval']

    ista_bounds = ista['glob_bestbound']
    fista_bounds = fista['glob_bestbound']

    sample_max_df = samples_to_max(samples_df)
    ista_samples = sample_max_df['ista']
    fista_samples = sample_max_df['fista']

    K_max = 7
    K_vals = range(1, K_max + 1)

    obj_m = 'v'
    bound_m = '^'
    samp_m = 'x'

    obj_color = 'c'
    bound_color = 'm'
    samp_color = 'r'

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
    ax0.set_xlabel('$K$')
    ax1.set_xlabel('$K$')
    ax0.set_ylabel('Worst case fixed-point residual')
    ax0.set_yscale('log')

    ax0.plot(K_vals, ista_objs[:K_max], marker=obj_m, color=obj_color, label='Best lower bound')
    ax0.plot(K_vals, ista_bounds[:K_max], marker=bound_m, color=bound_color, label='Best upper bound')
    ax0.plot(K_vals, ista_samples[:K_max], marker=samp_m, color=samp_color, label='SM')
    ax0.set_xticks(K_vals)
    ax0.set_title('ISTA')

    ax1.plot(K_vals, fista_objs[:K_max], marker=obj_m, color=obj_color)
    ax1.plot(K_vals, fista_bounds[:K_max], marker=bound_m, color=bound_color)
    ax1.plot(K_vals, fista_samples[:K_max], marker=samp_m, color=samp_color)
    ax1.set_xticks(K_vals)
    ax1.set_title('FISTA')

    fig.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 0.11))

    plt.suptitle(r'Lasso Exact VP Results')
    plt.tight_layout()

    axes = [ax0, ax1]
    for ax in axes:
        print(ax.get_position())
        pos = ax.get_position()
        new_pos = [pos.x0, pos.y0+.05, pos.width, pos.height]
        ax.set_position(new_pos)

    # plt.show()
    plt.savefig('plots/ista_fista_glob.pdf')


def main():
    sdp_df = pd.read_csv('data/ISTA_sublinconv.csv')
    samples_df = pd.read_csv('data/samples.csv')
    pep_df = pd.read_csv('data/pep.csv')

    plot_resids(sdp_df, pep_df, samples_df)
    plot_global_ista(samples_df)


if __name__ == '__main__':
    main()
