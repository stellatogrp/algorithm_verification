import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as spa
from UQP_class import UnconstrainedQuadraticProgram

from algocert.solvers.sdp_custom_solver.psd_cone_handler import PSDConeHandler
from algocert.solvers.sdp_custom_solver.solve_via_mosek import solve_via_mosek


def sample_point(n, r):
    # sample point in R^n that is l2 distance r away from 0
    samp = np.random.randn(n)
    samp = r * samp / np.linalg.norm(samp)
    return samp


def run_single_sdp(C, c, r):
    n = C.shape[0]
    n + 1

    Aeq = np.zeros((n + 1, n + 1))
    Aeq[-1, -1] = 1
    Aeq = spa.csc_matrix(Aeq)

    A = np.eye(n + 1)
    A[-1, :n] = -c
    A[:n, -1] = -c
    A[-1, -1] = 0
    A = spa.csc_matrix(A)

    A_vals = [Aeq, A]
    b_lvals = [1, 0]
    b_uvals = [1, r ** 2 - c.T @ c]

    C_mat = np.zeros((n + 1, n + 1))
    C_mat[:n, :n] = C
    C_mat = spa.csc_matrix(C_mat)

    psd_cone_handler = PSDConeHandler((0, n + 1))
    solve_via_mosek(C_mat, A_vals, b_lvals, b_uvals, [psd_cone_handler], n + 1)
    return 0


def run_sdp_cvxpy(C, c, r):
    n = C.shape[0]
    X = cp.Variable((n, n), symmetric=True)
    x = cp.Variable(n)
    x_mat = cp.reshape(x, (n, 1))

    obj = cp.Minimize(cp.trace(C @ X))
    constraints = [
        cp.trace(X) <= r ** 2 + 2 * c.T @ x - c.T @ c,
        cp.bmat([
            [X, x_mat],
            [x_mat.T, np.array([[1]])]
        ]) >> 0,
    ]
    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=False)
    # print(-res)
    return -res


def run_sdp_cert_probs(x0, P, t, ball_r, K_max=10):
    n = P.shape[0]
    ItP = np.eye(n) - t * P
    out = []
    for K in range(K_max):
        Ckplus1 = np.linalg.matrix_power(ItP, K+1)
        Ck = np.linalg.matrix_power(ItP, K)
        B = Ck - Ckplus1
        C_obj = - B.T @ B
        # sdp_obj = run_single_sdp(C_obj, x0, ball_r)
        sdp_obj = run_sdp_cvxpy(C_obj, x0, ball_r)
        out.append(sdp_obj)
    return out


def uqp_experiment():
    n = 25
    r = 10
    ball_r = 1
    N = 4
    mu = 1
    L = 100
    t = 2 / (mu + L)
    K_max = 10
    uqp = UnconstrainedQuadraticProgram(n, mu=mu, L=L)
    outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    out_fname = outf_prefix + 'paper_experiments/UQP/data/samples.csv'
    print('eigvals of P:', np.round(np.linalg.eigvals(uqp.P), 4))

    samples = []
    for _ in range(N):
        samples.append(sample_point(n, r))

    results = []
    for i, samp in enumerate(samples):
        sdp_objs = run_sdp_cert_probs(samp, uqp.P, t, ball_r, K_max=K_max)
        print(i, sdp_objs)
        for K in range(K_max):
            out_dict = dict(
                mu=mu,
                L=L,
                t=t,
                n=n,
                sample=i,
                K=K+1,
                sdp_obj=sdp_objs[K]
            )
            results.append(pd.Series(out_dict))

    res_df = pd.DataFrame(results)
    print(res_df)
    res_df.to_csv(out_fname, index=False)


def main():
    uqp_experiment()


if __name__ == '__main__':
    main()
