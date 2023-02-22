import matplotlib.pyplot as plt
import pandas as pd

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})

def plot_results(sdp_df, global_df, filename):
    """
    assumes the sdp_df and global_df each take the form
        num_iter       obj  solve_time
    0       1.0  4.083796    2.716526
    1       2.0  3.571718   12.727920
    2       3.0  4.339577   22.580774
    3       4.0  7.054666   33.536669
    """

    # plot the objective values
    plt.plot(sdp_df['num_iter'], sdp_df['obj'], label='sdp')
    plt.plot(global_df['num_iter'], global_df['obj'], label='global')
    plt.title('Convergence residuals')
    plt.xlabel('$K$')
    plt.ylabel('maximum $||z^K - z^{K-1}||_2^2$')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"{filename}_objvals.pdf")
    plt.clf()

    # plot the solution times
    plt.plot(sdp_df['num_iter'], sdp_df['solve_time'], label='sdp')
    plt.plot(global_df['num_iter'], global_df['solve_time'], label='global')
    plt.title('Solve times')
    plt.xlabel('$K$')
    plt.ylabel('maximum $||z^K - z^{K-1}||_2^2$')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"{filename}_solve_times.pdf")
    plt.clf()