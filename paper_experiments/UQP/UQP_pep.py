import numpy as np
import pandas as pd
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from UQP_class import UnconstrainedQuadraticProgram


def pep_single(mu, L, t, r, K=1):
    verbose=1
    problem = PEP()
    print(mu, L)

    # Declare a convex and a smooth convex function.
    func = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    # fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Compute n steps of the Douglas-Rachford splitting starting from x0
    x = [x0 for _ in range(K + 1)]
    for i in range(K):
        x[i+1] = x[i] - t * func.gradient(x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)
    # problem.set_initial_condition((x[1] - x[0]) ** 2 <= r ** 2)

    problem.set_performance_metric((x[-1] - x[-2]) ** 2)
    # problem.set_performance_metric((x[-1] - xs) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    return pepit_tau


def uqp_experiment():
    n = 25
    r = 10
    ball_r = 1
    mu = 1
    L = 100
    K_max = 10
    t = 2 / (mu + L)
    uqp = UnconstrainedQuadraticProgram(n, mu=mu, L=L)
    outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    out_fname = outf_prefix + 'paper_experiments/UQP/data/pep.csv'
    print('eigvals of P:', np.round(np.linalg.eigvals(uqp.P), 4))

    results = []
    for K in range(1, K_max + 1):
        tau = pep_single(mu, L, t, r + ball_r, K=K)
        out_dict = dict(
            L=L,
            mu=mu,
            t=t,
            K=K,
            pep_bound=tau,
        )
        results.append(pd.Series(out_dict))
    res_df = pd.DataFrame(results)
    print(res_df)

    res_df.to_csv(out_fname, index=False)


def main():
    uqp_experiment()


if __name__ == '__main__':
    main()
