import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as spa
from scipy.stats import ortho_group

from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep

# from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import \
# NonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.init_set.box_set import BoxSet

# from algocert.init_set.box_stack_set import BoxStackSet
# from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
from algocert.init_set.const_set import ConstSet

# from algocert.init_set.control_example_set import ControlExampleSet
# from algocert.init_set.init_set import InitSet
# from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.objectives.outer_prod_trace import OuterProdTrace
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter

# from tqdm import tqdm, trange


def generate_problem(n, L=1, mu=100):
    Q = ortho_group.rvs(n)
    P_eigs = np.zeros(n)
    mid_eigs = np.random.uniform(low=L, high=mu, size=n-2)
    len_mid = mid_eigs.shape[0]
    P_eigs[0] = L
    P_eigs[1: 1 + len_mid] = mid_eigs
    P_eigs[-1] = mu
    P = Q @ np.diag(P_eigs) @ Q.T
    # P += 100
    print('eigvals of P:', np.round(np.linalg.eigvals(P), 4))
    print('min, max of P:', np.min(P), np.max(P))

    A = np.random.randn(n, n)
    ATA = A.T @ A
    print('eigvals of ATA:', np.round(np.linalg.eigvals(ATA), 4))
    print('eigvals of P + ATA:', np.round(np.linalg.eigvals(P + ATA), 4))
    x = np.random.randn(n)
    c = A @ x

    return P, A, c


def generate_rho_opt(P, A):
    # eps = 1e-6
    Pinv = np.linalg.inv(P)
    APinvAT = A @ Pinv @ A.T
    eigs = np.linalg.eigvals(APinvAT)
    print('eigs of APinvAT:', np.round(eigs, 4))
    lambd_max = np.max(eigs)
    lambd_min = np.min(eigs)
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


def OSQP_cert_prob(P, A, c, rho, q_l, q_u, K=1, solver="GLOBAL", minimize=False):
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
    prim = Iterate(m, name='primal')
    s5_D = Im
    s5_A = spa.bmat([[A, -Im]])
    s5_b = zeros_m
    step5 = HighLevelLinearStep(prim, [x, z], D=s5_D, A=s5_A, b=s5_b, Dinv=s5_D)

    # step 6
    dual = Iterate(n, name='dual')
    s6_D = In
    s6_A = spa.bmat([[P, A.T, In]])
    s6_b = zeros_n
    step6 = HighLevelLinearStep(dual, [x, mu, q], D=s6_D, A=s6_A, b=s6_b, Dinv=s6_D)

    steps = [step1, step2, step3, step4, step5, step6]

    initsets = [ConstSet(x, zeros_n), ConstSet(mu, zeros_m), ConstSet(
        z, zeros_m), ConstSet(prim, zeros_m), ConstSet(dual, zeros_m)]
    # paramsets = [BoxSet(q, zeros_n, 2 * ones_n)]
    paramsets = [BoxSet(q, q_l, q_u)]

    # obj = [ConvergenceResidual(x)]
    obj = [OuterProdTrace(prim), OuterProdTrace(dual)]
    # obj = [OuterProdTrace(prim)]
    # obj = [OuterProdTrace(dual)]

    CP = CertificationProblem(K, initsets, paramsets, obj, steps)
    # CP.print_cp()

    if solver == "GLOBAL":
        resg = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600, minimize=minimize)
        # print('global', resg)
        b_val = CP.get_param_map()
        return resg, b_val[q]
    if solver == "SDP":
        res = CP.solve(solver_type='SDP', add_RLT=True, verbose=True,  solver=cp.MOSEK, minimize=minimize)
        # print('sdp', res)
        return res


def experiment_g():
    np.random.seed(4)
    n = 2
    P, A, c = generate_problem(n)
    rho = generate_rho_opt(P, A)

    rho_opt = generate_rho_opt(P, A)
    print(rho_opt)
    rho_vals = [1, 5, np.round(rho_opt, 4), 10, 25]
    K_vals = [2, 3, 4, 5]
    # K_vals = [2]

    save_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/rho_opt/data/'
    fname = save_dir + 'PD_QP.csv'
    res_rows = []
    for K in K_vals:
        for rho in rho_vals:
            res_g = OSQP_cert_prob(P, A, c, rho, np.zeros((n, 1)), 2 * np.ones((n, 1)), K=K, solver="GLOBAL")
            res_sdp = OSQP_cert_prob(P, A, c, rho, np.zeros((n, 1)), 2 * np.ones((n, 1)), K=K, solver="SDP")
            res_row = pd.Series(
                {
                    'K': K,
                    'rho': rho,
                    'g_obj': res_g[0],
                    'g_solve_time': res_g[1],
                    'sdp_obj': res_sdp[0],
                    'sdp_solve_time': res_sdp[1],
                    'min_max': 'max',
                }
            )
            res_rows.append(res_row)

            res_g = OSQP_cert_prob(P, A, c, rho, np.zeros((n, 1)), 2 * np.ones((n, 1)), K=K,
                                   solver="GLOBAL", minimize=True)
            res_sdp = OSQP_cert_prob(P, A, c, rho, np.zeros((n, 1)), 2 * np.ones((n, 1)), K=K,
                                     solver="SDP", minimize=True)
            res_row = pd.Series(
                {
                    'K': K,
                    'rho': rho,
                    'g_obj': res_g[0],
                    'g_solve_time': res_g[1],
                    'sdp_obj': res_sdp[0],
                    'sdp_solve_time': res_sdp[1],
                    'min_max': 'min',
                }
            )
            res_rows.append(res_row)
            df = pd.DataFrame(res_rows)
            df.to_csv(fname, index=False)
    print(df)


def main():
    np.random.seed(4)
    n = 2
    P, A, c = generate_problem(n)
    rho = generate_rho_opt(P, A)
    # print('rho opt:', rho)
    # rho = 25
    test_with_cvxpy(P, A, c)

    res_g, b_val = OSQP_cert_prob(P, A, c, rho, np.zeros((n, 1)), 1 * np.ones((n, 1)),
                                  K=2, solver="GLOBAL", minimize=False)
    # res_s = OSQP_cert_prob(P, A, c, rho, np.zeros((n, 1)), 1 * np.ones((n, 1)), K=3, solver="SDP", minimize=False)
    print('g:', res_g)
    print(b_val)
    # print('sdp:', res_s)

    # experiment_g()


if __name__ == '__main__':
    main()
