import numpy as np
import cvxpy as cp
import os
import joblib
import pandas as pd

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction, ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step

from scipy.stats import ortho_group


def get_n_processes(max_n=np.inf):
    try:
        # Check number of cpus if we are on a SLURM server
        n_cpus = int(os.environ["SLURM_CPUS_PER_TASK"])
    except KeyError:
        n_cpus = joblib.cpu_count()
    n_proc = max(min(max_n, n_cpus), 1)
    return n_proc


def solve_single_NNLS_via_cvxpy(A, b):
    n = A.shape[1]
    x = cp.Variable(n)
    obj = .5 * cp.sum_squares(A @ x - b)
    constraints = [x >= 0]
    problem = cp.Problem(cp.Minimize(obj), constraints)
    res = problem.solve()
    return res, x.value


def run_single_prox_grad_descent(tATA, tAtb, num_iter=5, x_init=None):
    def nonneg_proj(x):
        return np.maximum(x, 0)
    n = tATA.shape[0]
    In = np.eye(n)
    C = In - tATA
    if x_init is None:
        x_init = np.zeros(n)
    iterates = [x_init]
    xk = x_init
    for i in range(num_iter):
        xk = nonneg_proj(C @ xk + tAtb)
        iterates.append(xk)

    conv_resid = np.linalg.norm(iterates[-1] - iterates[-2])
    return conv_resid ** 2, iterates[-1]


def theoretical_PEPit_val(L, mu, t, r, N=1):
    t = 2 / (mu + L)
    theoretical_tau = max((1 - mu*t)**2, (1 - L*t)**2)**N
    print(L, mu, theoretical_tau)
    return theoretical_tau * (r ** 2)


def test_PEPit_val(L, mu, t, r, N=1):
    problem = PEP()
    verbose = 0

    # Declare a strongly convex smooth function and a closed convex proper function
    f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    f2 = problem.declare_function(ConvexFunction)
    func = f1 + f2

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)

    # Run the proximal gradient method starting from x0
    x = x0
    x_vals = [x0]
    for _ in range(N):
        y = x - t * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, t)
        x_vals.append(x)

    # Set the performance metric to the distance between x and xs
    problem.set_performance_metric((x_vals[-1] - x_vals[-2]) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau


def generate_single_A(m, n, L, mu):
    sq_mu = np.sqrt(mu)
    sq_L = np.sqrt(L)
    spectrum = np.random.uniform(sq_mu, sq_L, size=n-2)
    # print(spectrum)
    P = np.zeros((m, n))
    # P[:n, :n] = np.diag(spectrum)
    for i in range(1, n-1):
        P[i, i] = spectrum[i-2]
    P[0, 0] = sq_L
    P[n-1, n-1] = sq_mu
    # print(P)
    Um = ortho_group.rvs(m)
    Un = ortho_group.rvs(n)
    # print(Um @ P @ Un)
    A = Um @ P @ Un
    # print(Um.shape, P.shape, Un.shape)
    ATA = A.T @ A
    eigvals = np.linalg.eigvals(ATA)
    # print(np.min(eigvals), np.max(eigvals))
    return A


def generate_A_matrices(n_vals, m_vals, L, mu):
    out = []
    for (n, m) in zip(n_vals, m_vals):
        # A = np.random.randn(m, n)
        A = generate_single_A(m, n, L, mu)
        out.append(A)
    return out


def generate_sample_b(m):
    return 10 * np.random.rand(m) + 10


def single_sample_experiment(A, t, num_iter, warm_start_x):
    (m, n) = A.shape
    ATA = A.T @ A
    b = generate_sample_b(m)
    ATb = A.T @ b

    res, x = solve_single_NNLS_via_cvxpy(A, b)
    prox_res, prox_x = run_single_prox_grad_descent(t * ATA, t * ATb, num_iter=num_iter)
    ws_prox_res, ws_prox_x = run_single_prox_grad_descent(t * ATA, t * ATb, num_iter=num_iter, x_init=warm_start_x)

    sample_r = np.linalg.norm(x)
    return prox_res, sample_r, ws_prox_res


def single_n_experiment(n_vals, m_vals, A_matrices, experiment_number, num_iter=5, num_samples=10):
    try:
        n = n_vals[experiment_number]
        m = m_vals[experiment_number]
        A = A_matrices[experiment_number]
        max_num_jobs = get_n_processes(max_n=num_samples)
        ATA = A.T @ A
        eigvals = np.linalg.eigvals(ATA)
        mu = np.min(eigvals)
        L = np.max(eigvals)
        print(mu, L)
        t = 2 / (mu + L)

        # create warm started x init
        warm_start_b = generate_sample_b(m)
        _, warm_start_x = solve_single_NNLS_via_cvxpy(A, warm_start_b)

        results = joblib.Parallel(n_jobs=max_num_jobs)(joblib.delayed(single_sample_experiment)
                                                       (A, t, num_iter, warm_start_x) for _ in range(num_samples))

        # print(results)
        conv_resids = [r[0] for r in results]
        max_test_r = max(results, key=lambda val: val[1])[1]
        warm_start_conv_resids = [r[2] for r in results]

        tau = test_PEPit_val(L, mu, t, max_test_r, N=5)
        return conv_resids, warm_start_conv_resids, max_test_r, tau
    except Exception as e:
        print(f'failure for n={n}')
        print(e)
        return None, 0, 0


def test_many_n_samples():
    np.random.seed(0)

    mu = 1
    L = 10
    n_vals = np.array([5, 10, 25, 50, 75, 100, 150, 200, 300])
    n_vals = np.array([5, 10, 25])
    m_vals = 2 * n_vals
    num_samples = 10
    num_iter = 5

    njobs = get_n_processes(max_n=20)
    A_matrices = generate_A_matrices(n_vals, m_vals, L, mu)

    num_experiments = len(n_vals)

    results = joblib.Parallel(n_jobs=njobs)(joblib.delayed(single_n_experiment)
                                            (n_vals, m_vals, A_matrices, r, num_iter=num_iter, num_samples=num_samples)
                                            for r in range(num_experiments)
                                            )
    sample_series = []
    pep_series = []
    for i in range(num_experiments):
        n = n_vals[i]
        m = m_vals[i]
        conv_resids, warm_start_conv_resids, max_test_r, pepit_max_obj = results[i]
        print(conv_resids, max_test_r, pepit_max_obj)
        if conv_resids is not None:
            new_pep_row = pd.Series(
                {
                    'n': n,
                    'm': m,
                    'max_test_r': max_test_r,
                    'pepit_max_sample_obj': pepit_max_obj,
                }
            )
            pep_series.append(new_pep_row)
            for j in range(num_samples):
                new_sample_row = pd.Series(
                    {
                        'n': n,
                        'm': m,
                        'num_prox_grad_iter': num_iter,
                        'conv_resid': conv_resids[j],
                        'warm_start_conv_resid': warm_start_conv_resids[j],
                    }
                )
                sample_series.append(new_sample_row)
                # print(new_row)
                # df_samples = pd.concat([df_samples, new_row], axis=0, ignore_index=True)
        df_samples = pd.DataFrame(sample_series)
    print(df_samples)
    df_pep = pd.DataFrame(pep_series)
    print(df_pep)

    out_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/param_effect/data/'
    # out_dir = '/home/vranjan/experiments/param_effect/data/'

    sample_out_fname = out_dir + 'test_out.csv'
    pep_out_fname = out_dir + 'test_pep.csv'

    df_samples.to_csv(sample_out_fname, index=False)
    df_pep.to_csv(pep_out_fname, index=False)


def main():
    test_many_n_samples()


if __name__ == '__main__':
    main()
