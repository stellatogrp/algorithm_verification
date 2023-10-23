
import cvxpy as cp
import numpy as np
import pandas as pd
from NUM_class import NUM

# from PEPit.examples.composite_convex_minimization.proximal_gradient import wc_proximal_gradient
from PEPit import PEP
from PEPit.functions import ConvexFunction, SmoothStronglyConvexFunction
from PEPit.primitive_steps import proximal_step


def run_DR_single(M, q, z0, Pi, K):
    out_zres = []
    lhs = M + np.eye(M.shape[0])
    zk = z0
    for _ in range(K):
        ukplus1 = np.linalg.solve(lhs, zk - q)
        utilde_kplus1 = Pi(2 * ukplus1 - zk)
        zkplus1 = zk + utilde_kplus1 - ukplus1
        out_zres.append(np.linalg.norm(zkplus1 - zk) ** 2)
        zk = zkplus1
    return out_zres


def run_DR_cs_ws(M, q, z_ws, Pi, K_max=5):
    # out_cs = []
    # out_ws = []

    z_cs = np.zeros(M.shape[0])
    out_cs = run_DR_single(M, q, z_cs, Pi, K_max)
    out_ws = run_DR_single(M, q, z_ws, Pi, K_max)

    return out_cs, out_ws


def sample_l2ball(c, r, N):
    all_samples = []
    for _ in range(N):
        sample = np.random.normal(0, 1, c.shape[0])
        sample = np.random.uniform(0, r) * sample / np.linalg.norm(sample)
        # print(np.linalg.norm(sample))
        # print(sample.reshape(-1, 1) + c)
        all_samples.append(sample.reshape(-1, 1) + c)
    return all_samples


def c_to_xopt(instance, c_vals):
    out = []
    A = instance.A
    m, n = A.shape
    w = instance.w

    for c in c_vals:
        f = cp.Variable(n)
        rhs = np.hstack([c.reshape(-1, ), np.zeros(n), instance.t])
        obj = cp.Minimize(w @ f)
        constraints = [A @ f <= rhs]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        z_out = np.hstack([f.value, constraints[0].dual_value])
        out.append(z_out)
        # print(c, np.round(z_out, 4))

    return out


def c_to_DR(instance, c_vals, z_ws, K=5):
    M = instance.M
    M.shape[0]
    n = instance.A.shape[1]
    w = instance.w
    z_ws = z_ws.reshape(-1, )

    def Pi(x):
        m, n = instance.A.shape
        ret = x.copy()
        ret[n:] = np.maximum(ret[n:], 0)
        return ret

    # x = np.random.normal(size=15)
    # print(x, Pi(x))

    out_series = []
    for i, c in enumerate(c_vals):
        q = np.hstack([w, c.reshape(-1, ), np.zeros(n), instance.t])
        out_cs, out_ws = run_DR_cs_ws(M, q, z_ws, Pi, K_max=K)
        print('--')
        print(out_cs)
        print(out_ws)

        for K_val in range(K):
            cs_res = out_cs[K_val]
            ws_res = out_ws[K_val]
            out_dict_cs = dict(type='cs', sample_num=(i + 1), K=(K_val + 1), res=cs_res)
            out_dict_ws = dict(type='ws', sample_num=(i + 1), K=(K_val + 1), res=ws_res)
            out_series += [pd.Series(out_dict_cs), pd.Series(out_dict_ws)]
    out_df = pd.DataFrame(out_series)
    out_df.to_csv('data/num_sample.csv', index=False)


def DR_pep(K, r, L=1, alpha=1, theta=1):
    problem = PEP()

    # func1 = problem.declare_function(SmoothConvexFunction, L=L)
    # func1 = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=.1)
    # func2 = problem.declare_function(ConvexFunction)
    # func = func1 + func2

    # zs = func.stationary_point()
    # # ws = func.stationary_point()

    # z0 = problem.set_initial_point()
    # w0 = problem.set_initial_point()

    # z = [z0 for _ in range(K + 1)]
    # w = [w0 for _ in range(K)]

    # for i in range(K):
    #     w[i], _, _ = proximal_step(z[i], func1, alpha)
    #     y, _, _ = proximal_step(2 * w[i] - z[i], func2, alpha)
    #     z[i + 1] = z[i] + theta * (y - w[i])

    # problem.set_initial_condition((z[0] - zs) ** 2 <= r ** 2)
    # problem.set_initial_condition(w0 ** 2 <= 0)

    # problem.set_performance_metric((z[-1] - z[-2]) ** 2)

    # pepit_tau = problem.solve(verbose=1)

    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=0)
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    # fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Compute n steps of the Douglas-Rachford splitting starting from x0
    x = [x0 for _ in range(K)]
    w = [x0 for _ in range(K + 1)]
    for i in range(K):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x[0] - xs) ** 2 <= r ** 2)

    # Set the performance metric to the final distance to the optimum in function values
    # problem.set_performance_metric((func2(y) + fy) - fs)
    problem.set_performance_metric((w[-1] - w[-2]) ** 2)

    pepit_tau = problem.solve(verbose=1)
    return pepit_tau


def r_to_pep(instance, r_cs, r_ws, K=5):
    M = instance.M
    MpI = M + np.eye(M.shape[0])
    eigs = np.linalg.eigvals(MpI)
    print(r_cs, r_ws)
    # print(eigs)
    L = np.max(np.abs(eigs))

    out_series = []
    for K_val in range(1, K+1):
        cs_tau = DR_pep(K_val, r_cs, L=L)
        ws_tau = DR_pep(K_val, r_ws, L=L)
        print(cs_tau, ws_tau)
        cs_outdict = dict(type='cs', K=K_val, tau=cs_tau)
        ws_outdict = dict(type='ws', K=K_val, tau=ws_tau)
        out_series += [pd.Series(cs_outdict), pd.Series(ws_outdict)]
    out_df = pd.DataFrame(out_series)
    print(out_df)
    out_df.to_csv('data/num_pep.csv', index=False)


def sample_and_run(instance, c_c, c_r, N, K=5):
    c_vals = sample_l2ball(c_c, c_r, N)
    # print(c_vals)
    x_opt_vals = c_to_xopt(instance, c_vals)
    z_ws = instance.test_cp_prob().reshape(-1, )
    x_maxcs = max(x_opt_vals, key=lambda x: np.linalg.norm(x))
    x_rcs = np.linalg.norm(x_maxcs)
    x_maxws = max(x_opt_vals, key=lambda x: np.linalg.norm(x - z_ws))
    x_rws = np.linalg.norm(x_maxws - z_ws)
    # print(x_rcs, x_rws)

    # c_to_DR(instance, c_vals, z_ws, K=K)
    r_to_pep(instance, x_rcs, x_rws, K=K)


def main():
    m, n = 3, 4
    c_c = 2 * np.ones((m, 1))
    c_r = .1

    instance = NUM(m, n, c_c, c_r=c_r, seed=3)
    # print(instance.test_cp_prob())

    N = 100
    np.random.seed(0)
    sample_and_run(instance, c_c, c_r, N)


if __name__ == '__main__':
    main()
