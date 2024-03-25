from datetime import datetime

import cvxpy as cp
import numpy as np
import pandas as pd
from car2D import Car2D
from MPC_experiment import simulate_steps
from PEPit import PEP
from PEPit.functions import (
    ConvexFunction,
    SmoothStronglyConvexQuadraticFunction,
)
from PEPit.primitive_steps import proximal_step
from tqdm import tqdm


def OSQP(car, P, A, l, u, x0, sigma=1e-6, K=6, rho_const=True):
    m, n = A.shape
    if rho_const:
        rho = np.eye(m)
    else:
        eq_idx = list(range(car.nx))
        rho = np.ones(m)
        rho[eq_idx] *= 10
        rho = np.diag(rho)
    rho_inv = np.linalg.inv(rho)

    xk = x0.copy()
    zk = np.zeros(m)
    yk = np.zeros(m)
    sk = np.zeros(m)

    def Pi(x):
        return np.minimum(np.maximum(x, l.reshape(-1, )), u.reshape(-1, ))

    out_res = []
    lhs = P + sigma * np.eye(n) + A.T @ rho @ A
    for _ in range(K):
        # sigma x + A^T rho z - A^T y
        rhs = sigma * xk + A.T @ rho @ zk - A.T @ yk
        xnew = np.linalg.solve(lhs, rhs)
        znew = Pi(A @ xnew + rho_inv @ yk)
        ynew = yk + rho @ (A @ xnew - znew)
        snew = znew + rho_inv @ ynew

        # print(xnew.shape, znew.shape, ynew.shape, snew.shape)

        out_res.append(np.linalg.norm(xnew - xk) ** 2 + np.linalg.norm(snew - sk) ** 2)

        xk = xnew
        zk = znew
        yk = ynew
        sk = snew
    # print(out_res)

    return out_res


def single_MPC_run(car, xinit, uinit, x0, K=7):
    H, M, l1, l2, u1, u2 = car.get_QP_data()

    l = np.hstack([xinit, l1, -car.smax + uinit, l2])
    u = np.hstack([xinit, u1, car.smax + uinit, u2])

    rhoconst_res = OSQP(car, H, M, l, u, x0, K=K, rho_const=True)
    rhoadj_res = OSQP(car, H, M, l, u, x0, K=K, rho_const=False)

    return rhoconst_res, rhoadj_res


def compute_max_r(car, xinit_samples, uinit_samples, shifted_sols):
    opt_sols = []
    for xinit, uinit in zip(xinit_samples, uinit_samples):
        opt_sols.append(car.solve_via_cvxpy(xinit, uinit=uinit))

    max_rvals = []
    # shifted_sols = [np.zeros(12)]

    H, M, l1, l2, u1, u2 = car.get_QP_data()

    def max_fun(opt, x0):
        x_diff = np.linalg.norm(opt)
        # z_diff = np.linalg.norm(M @ (opt - x0))
        # print(x_diff, z_diff)
        z_diff = np.linalg.norm(M @ opt)
        return np.sqrt(x_diff ** 2 + z_diff ** 2)

    for x0 in tqdm(shifted_sols, desc='computing radii'):
        # maximizer = max(opt_sols, key=lambda opt: np.linalg.norm(opt - x0))
        maximizer = max(opt_sols, key=lambda opt: max_fun(opt, x0))
        max_r = max_fun(maximizer, x0)
        # print(max_r)
        max_rvals.append(max_r)
    return np.max(max_rvals)


def MPC_pep(car, r, K):
    print(K)

    problem = PEP()
    # L = 1
    # mu = 0
    # print(np.linalg.eigvals(car.H.todense()))
    # print(mu)
    L = np.max(np.linalg.eigvals(car.H.todense()))
    mu = np.min(np.linalg.eigvals(car.H.todense()))
    alpha = 1
    theta = 1


    func1 = problem.declare_function(ConvexFunction)
    # func2 = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    func2 = problem.declare_function(SmoothStronglyConvexQuadraticFunction, L=L, mu=mu)
    # func2 = problem.declare_function(ConvexFunction)
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    # fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    x = [x0 for _ in range(K)]
    w = [x0 for _ in range(K + 1)]
    for i in range(K):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x[0] - xs) ** 2 <= r ** 2)
    problem.set_initial_condition((w[0] - xs) ** 2 <= r ** 2)

    # Set the performance metric to the final distance to the optimum in function values
    # problem.set_performance_metric((func2(y) + fy) - fs)
    if K == 1:
        problem.set_performance_metric((x[-1] - x0) ** 2 + (w[-1] - w[-2]) ** 2)
    else:
        problem.set_performance_metric((x[-1] - x[-2]) ** 2 + (w[-1] - w[-2]) ** 2)

    mosek_params = {
        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-8,
        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-8,
        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-7,
    }
    pepit_tau = problem.solve(verbose=2, solver=cp.MOSEK, mosek_params=mosek_params)
    # pepit_tau = problem.solve(verbose=2, wrapper='mosek')
    print(pepit_tau)
    # exit(0)
    return pepit_tau


def MPC_samp_pep(outf, K_max=7, eps=1e-3):
    T = 5
    # N = 10000
    N = 100
    # N = 5

    np.random.seed(2)
    car = Car2D(T=T)

    # car, xinit_samples, uinit_samples, sol, shifted_sols = simulate_steps(T=T, N=N, eps=eps)
    xinit_samples, uinit_samples, sol, shifted_sols = simulate_steps(car, T=T, N=N, eps=eps)
    xinit_min = np.min(xinit_samples, axis=0)
    xinit_max = np.max(xinit_samples, axis=0)
    uinit_min = np.min(uinit_samples, axis=0)
    uinit_max = np.max(uinit_samples, axis=0)
    # K = 2
    print(xinit_min, xinit_max)
    print(uinit_min, uinit_max)

    x0_min = np.min(shifted_sols, axis=0)
    x0_max = np.max(shifted_sols, axis=0)
    print(x0_min, x0_max)

    max_r = compute_max_r(car, xinit_samples, uinit_samples, shifted_sols)
    print(max_r)

    const_res = []
    adj_res = []
    for xinit, uinit, x0 in zip(xinit_samples, uinit_samples, shifted_sols):
        # print(xinit, uinit, x0)
        rhoconst_res, rhoadj_res = single_MPC_run(car, xinit, uinit, x0)
        const_res.append(rhoconst_res)
        adj_res.append(rhoadj_res)

    pep_vals = []
    for K in range(1, K_max + 1):
        pep_vals.append(MPC_pep(car, max_r, K))

    print(np.max(const_res, axis=0))
    print(np.max(adj_res, axis=0))
    print(pep_vals)

    maxconstres = np.max(const_res, axis=0)
    maxadjres = np.max(adj_res, axis=0)

    # return np.max(const_res, axis=0), np.max(adj_res, axis=0)

    df = []
    for i in range(K_max):
        df.append(pd.Series({
            'K': i+1,
            'const_samp_max': maxconstres[i],
            'adj_samp_max': maxadjres[i],
            'const_pep': pep_vals[i],
        }))
    df = pd.DataFrame(df)
    print(df)

    df.to_csv(outf, index=False)


def main():
    d = datetime.now()
    # print(d)
    d.strftime('%m%d%y_%H%M%S')
    # outf_prefix = '/home/vranjan/algorithm-certification/'
    outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-verification/'
    outf = outf_prefix + 'paper_experiments/MPC/data/ws_samp_pep_quad.csv'
    print(outf)

    # simulate_steps()
    MPC_samp_pep(outf)


if __name__ == '__main__':
    main()
