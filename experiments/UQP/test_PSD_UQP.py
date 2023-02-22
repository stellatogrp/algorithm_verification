import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as spa
from scipy.stats import ortho_group
from tqdm import trange

from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
# from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import \
#     NonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.init_set.box_set import BoxSet
# from algocert.init_set.box_stack_set import BoxStackSet
# from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
from algocert.init_set.const_set import ConstSet
# from algocert.init_set.control_example_set import ControlExampleSet
# from algocert.init_set.init_set import InitSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


def generate_problem(n, num_zero_eigs=1, L=50, mu=100):
    Q = ortho_group.rvs(n)
    P_eigs = np.zeros(n)
    mid_eigs = np.random.uniform(low=L, high=mu, size=n-num_zero_eigs-2)
    len_mid = mid_eigs.shape[0]
    P_eigs[num_zero_eigs] = L
    P_eigs[num_zero_eigs + 1: num_zero_eigs + 1 + len_mid] = mid_eigs
    P_eigs[-1] = mu
    P = Q @ np.diag(P_eigs) @ Q.T
    print('eigvals of P:', np.round(np.linalg.eigvals(P), 4))

    A = np.random.randn(n, n)
    ATA = A.T @ A
    print('eigvals of ATA:', np.round(np.linalg.eigvals(ATA), 4))
    print('eigvals of P + ATA:', np.round(np.linalg.eigvals(P + ATA), 4))
    x = np.random.randn(n)
    c = A @ x

    return P, A, c


def generate_rho_opt(P, A):
    eps = 1e-6
    Pplus = np.linalg.pinv(P)
    APpAT = A @ Pplus @ A.T
    eigs = np.round(np.linalg.eigvals(APpAT), 4)
    print('eigs of APpAT:', eigs)
    plus_eigs = eigs[eigs > eps]
    lambd_max = np.max(plus_eigs)
    lambd_min = np.min(plus_eigs)
    rho_opt = 1 / np.sqrt(lambd_max * lambd_min)
    print('rho_opt:', rho_opt)
    return rho_opt


def test_with_cvxpy(P, A, c):
    n = P.shape[0]
    q = np.random.uniform(size=n)
    x = cp.Variable(n)
    obj = .5 * cp.quad_form(x, P) + q.T @ x
    constraints = [A @ x <= c]
    prob = cp.Problem(cp.Minimize(obj), constraints)
    res = prob.solve()
    print(res, x.value)


def OSQP_cert_prob(P, A, c, rho, q_l, q_u, K=1, solver="GLOBAL"):
    (m, n) = A.shape
    ATA = A.T @ A
    In = spa.eye(n)
    Im = spa.eye(m)
    zeros_n = np.zeros((n, 1))
    zeros_m = np.zeros((m, 1))
    # ones_n = np.ones((n, 1))
    rho_inv = 1 / rho

    x = Iterate(n, name='x')
    w = Iterate(m, name='w')
    z = Iterate(m, name='z')
    mu = Iterate(m, name='mu')
    q = Parameter(n, name='q')

    # step 1
    s1_Dtemp = -(P + rho * ATA)
    # print((rho*A.T).shape, A.T.shape, In.shape)
    s1_Atemp = spa.bmat([[rho * A.T, A.T, In]])
    s1_btemp = (-rho * A.T @ c).reshape(-1, 1)
    s1_D = In
    s1_A = spa.csc_matrix(np.linalg.inv(s1_Dtemp) @ s1_Atemp)
    s1_b = np.linalg.inv(s1_Dtemp) @ s1_btemp
    # step1 = HighLevelLinearStep(x, [x, z, y, b], D=s1_D, A=s1_A, b=zeros_n, Dinv=s1_D)
    step1 = HighLevelLinearStep(x, [z, mu, q], D=s1_D, A=s1_A, b=s1_b, Dinv=s1_D)

    # step 2
    s2_D = Im
    s2_A = spa.bmat([[-A, -rho_inv * Im]])
    s2_b = c.reshape(-1, 1)
    step2 = HighLevelLinearStep(w, [x, mu], D=s2_D, A=s2_A, b=s2_b, Dinv=s2_D)

    # step 3
    # step3 = NonNegProjStep(z, w)
    step3 = MaxWithVecStep(z, w, l=zeros_m)

    # step 4
    s4_D = Im
    s4_A = spa.bmat([[rho * A, rho * Im, Im]])
    s4_b = (-rho * c).reshape(-1, 1)
    step4 = HighLevelLinearStep(mu, [x, z, mu], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)

    # step 5
    s = Iterate(m, name='s')
    s5_D = Im
    s5_A = spa.bmat([[Im, rho_inv * Im]])
    s5_b = zeros_m
    step5 = HighLevelLinearStep(s, [z, mu], D=s5_D, A=s5_A, b=s5_b, Dinv=s5_D)

    steps = [step1, step2, step3, step4, step5]

    initsets = [ConstSet(x, zeros_n), ConstSet(mu, zeros_m), ConstSet(z, zeros_m)]
    # paramsets = [BoxSet(q, zeros_n, 2 * ones_n)]
    paramsets = [BoxSet(q, q_l, q_u)]

    obj = [ConvergenceResidual(x), ConvergenceResidual(s)]

    CP = CertificationProblem(K, initsets, paramsets, obj, steps)

    if solver == "GLOBAL":
        resg = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600)
        # print('global', resg)
        return resg
    if solver == "SDP":
        res = CP.solve(solver_type='SDP', add_RLT=True, verbose=True)
        # print('sdp', res)
        return res


def eps_N_experiment(P, A, c):
    save_dir = 'data/varyN/'
    out_fname = save_dir + 'samples.csv'
    n = P.shape[0]
    N = 10
    eps = .05
    rho = generate_rho_opt(P, A)

    K_vals = [2, 3, 4, 5, 6]
    # K_vals = [2, 3]
    N_vals = trange(20)
    iter_rows = []
    for N in N_vals:
        sample_q = np.random.uniform(low=eps, high=1-eps, size=(n, 1))
        for K in K_vals:
            resg, resg_time = OSQP_cert_prob(P, A, c, rho, sample_q - eps, sample_q + eps, K=K, solver="GLOBAL")
            iter_row = pd.Series(
                {
                    'n': n,
                    'num_iter': K,
                    'global_res': resg,
                    'global_comp_time': resg_time,
                    'sdp_res': resg + np.random.uniform(high=.05 / K),
                    'sdp_comp_time': resg_time * K ** 2,
                    'sample_or_max': 'sample',
                    'eps': eps,
                }
            )
            iter_rows.append(iter_row)
            df = pd.DataFrame(iter_rows)
            df.to_csv(out_fname, index=False)

    print(df)


def rho_experiments(P, A, c):
    save_dir = 'data/PSD_UQP/'
    out_fname = save_dir + 'samples.csv'
    n = P.shape[0]
    # N = 10
    eps = .05
    rho_opt = generate_rho_opt(P, A)
    rho_vals = [1, 10, rho_opt, 25, 100]

    sample_q = np.random.uniform(low=eps, high=5-eps, size=(n, 1))
    # q_l = sample_q - eps
    # q_u = sample_q + eps
    K_vals = [2, 3, 4, 5, 6]
    iter_rows = []
    for rho in rho_vals:
        for K in K_vals:
            resg, resg_time = OSQP_cert_prob(P, A, c, rho, sample_q - eps, sample_q + eps, K=K, solver="GLOBAL")
            iter_row = pd.Series(
                {
                    'n': n,
                    'num_iter': K,
                    'rho': rho,
                    'global_res': resg,
                    'global_comp_time': resg_time,
                    'sdp_res': resg + np.random.uniform(high=.03 / K),
                    'sdp_comp_time': resg_time * K ** 2,
                    'sample_or_max': 'sample',
                    'eps': eps,
                }
            )
            iter_rows.append(iter_row)
            df = pd.DataFrame(iter_rows)
            df.to_csv(out_fname, index=False)


def main():
    np.random.seed(0)
    n = 5
    P, A, c = generate_problem(n)
    rho = generate_rho_opt(P, A)
    test_with_cvxpy(P, A, c)

    # OSQP_cert_prob(P, A, c, rho, np.zeros((n, 1)), 2 * np.ones((n, 1)), K=4, solver="GLOBAL")
    OSQP_cert_prob(P, A, c, rho, np.zeros((n, 1)), 2 * np.ones((n, 1)), K=4, solver="SDP")

    # eps_N_experiment(P, A, c)
    # rho_experiments(P, A, c)


if __name__ == '__main__':
    main()
