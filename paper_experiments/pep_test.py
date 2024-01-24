import cvxpy as cp
import matplotlib.pyplot as plt
import pandas as pd
from PEPit import PEP
from PEPit.functions import (
    SmoothStronglyConvexFunction,
)

# Set up plot style
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.sans-serif": ["Helvetica Neue"],
    "font.size": 20,
    "figure.figsize": (9, 6)})


def single_PEP(mu, L, R, t, K, test_opt_dist=False):
    verbose=2
    problem = PEP()
    print(mu, L)
    func = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()
    x = [x0 for _ in range(K + 1)]
    for i in range(K):
        x[i + 1] = x[i] - t * func.gradient(x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= R ** 2)
    if test_opt_dist:
        problem.set_performance_metric((x[-1] - xs) ** 2)
    else:
        problem.set_performance_metric((x[-1] - x[-2]) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    mosek_params = {
        # 'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-5,
    }
    pepit_tau = problem.solve(verbose=pepit_verbose, solver=cp.MOSEK, mosek_params=mosek_params)

    return pepit_tau


def PEP_experiments(mu, L, R, t_vals, K_max):
    res = []
    for K in range(1, K_max + 1):
        for t in t_vals:
            tau = single_PEP(mu, L, R, t, K)
            res.append(pd.Series({
                'mu': mu,
                'L': L,
                't': t,
                'K': K,
                'tau': tau,
            }))
    res_df = pd.DataFrame(res)
    print(res_df)
    res_df.to_csv('pep_test.csv', index=False)
    return res_df


def plot_df(df, mu, L, K_max):
    t_vals = df['t'].unique()
    fig, ax = plt.subplots()
    for K in range(1, K_max + 1):
        dfK = df[df['K'] == K]
        print(dfK)

        plt.plot(t_vals, dfK['tau'], label=f'K={K}')

    ax.set_yscale('log')
    ax.set_ylabel('PEP bound')
    ax.set_xlabel(r'$K$')
    ax.set_title(f'$\mu$ = {mu}, $L$ = {L}')
    ax.axvline(x=(2 / (mu + L)))

    plt.legend()
    # plt.show()
    plt.savefig('pep_test.pdf')


def main():
    mu = 20
    L = 100
    R = 1
    K_max = 7

    t_vals = [.01, .0125, .015, 2/ (mu + L), .0175, .02]
    t_vals.sort()
    df = PEP_experiments(mu, L, R, t_vals, K_max)

    plot_df(df, mu, L, K_max)


if __name__ == '__main__':
    main()
