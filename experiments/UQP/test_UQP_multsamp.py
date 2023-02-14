# import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as spa
from scipy.stats import ortho_group

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


def random_mat_cond_number(n, cond_P=10**2):
    '''
    Generates an n x n psd matrix with predefined condition number
    '''
    log_cond_P = np.log(cond_P)
    exp_vec = np.arange(-log_cond_P / 4., log_cond_P * (n + 1) / (4 * (n - 1)), log_cond_P / (2. * (n - 1)))
    s = np.exp(exp_vec)
    S = np.diag(s)
    Q = ortho_group.rvs(n)
    return Q @ S @ Q.T


def UQP(n, P, K=1, eps_b=.1, rho=None, b_sample=None, solver_type='SDP', add_RLT=True, verbose=False):

    delta = 10
    In = spa.eye(n)
    zeros_n = np.zeros((n, 1))

    # P = random_mat_cond_number(n, cond_P=1e2)
    # eigvals = np.linalg.eigvals(P)
    # print(np.max(eigvals) / np.min(eigvals))

    if rho is None:
        P_eigvals = np.linalg.eigvals(P)
        lambd_min = np.min(P_eigvals)
        lambd_max = np.max(P_eigvals)
        print(lambd_min, lambd_max)
        if delta < lambd_min:
            rho = np.sqrt(delta * lambd_min)
        elif delta > lambd_max:
            rho = np.sqrt(delta * lambd_max)
        else:
            rho = delta

    print('using rho =', rho)

    dp = delta + rho
    dp_inv = 1 / dp
    PpI = P + rho * In
    PpI_inv = spa.csc_matrix(np.linalg.inv(PpI))
    y = Iterate(n, name='y')
    w = Iterate(n, name='w')
    z = Iterate(n, name='z')
    q = Parameter(n, name='q')

    # step 1
    s1D = In
    s1A = PpI_inv
    step1 = HighLevelLinearStep(w, [z], D=s1D, A=s1A, b=zeros_n, Dinv=s1D)

    # step 2
    s2D = In
    s2A = PpI_inv
    step2 = HighLevelLinearStep(y, [q], D=s2D, A=s2A, b=zeros_n, Dinv=s2D)

    # step 3
    s3D = In
    s3A = spa.bmat([[delta * dp_inv * In, rho * (rho - delta) * In, -rho * dp_inv * In]])
    step3 = HighLevelLinearStep(z, [z, w, y], D=s3D, A=s3A, b=zeros_n, Dinv=s3D)

    steps = [step1, step2, step3]

    zset = ConstSet(z, np.zeros((n, 1)))

    # qset = ConstSet(q, np.ones((n, 1)))
    l = (1 - eps_b) * np.ones((n, 1))
    u = (1 + eps_b) * np.ones((n, 1))
    qset = BoxSet(q, l, u)

    obj = [ConvergenceResidual(z)]

    CP = CertificationProblem(K, [zset], [qset], obj, steps)

    # CP.canonicalize(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600, verbose=verbose)
    if solver_type == 'SDP':
        res = CP.solve(solver_type='SDP', add_RLT=add_RLT, verbose=verbose)
    if solver_type == 'GLOBAL':
        res = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600, verbose=verbose)
    return res


def experiment():

    save_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/UQP/data/'
    n = 5
    np.random.seed(0)
    Phalf = np.random.randn(n, n)
    P = Phalf.T @ Phalf

    # K_vals = range(1, 11)
    # eps_vals = [.01, .02, .1, .25, .5]
    K_vals = range(1, 10)
    eps_vals = [.01, .02, .1, .25, .5]
    res_srlt_rows = []
    res_g_rows = []
    for K in K_vals:
        for eps in eps_vals:
            print('K:', K, 'eps:', eps)
            res_srlt = UQP(n, P, eps_b=eps, K=K, solver_type='SDP')
            res_g = UQP(n, P, eps_b=eps, K=K, solver_type='GLOBAL')
            res_srlt_row = pd.Series(
                {
                    'num_iter': K,
                    'eps_b': eps,
                    'obj': res_srlt[0],
                    'solve_time': res_srlt[1],
                }
            )
            res_srlt_rows.append(res_srlt_row)

            res_g_row = pd.Series(
                {
                    'num_iter': K,
                    'eps_b': eps,
                    'obj': res_g[0],
                    'solve_time': res_g[1],
                }
            )
            res_g_rows.append(res_g_row)

    # df_s = pd.DataFrame(res_s_rows)
    df_srlt = pd.DataFrame(res_srlt_rows)
    df_g = pd.DataFrame(res_g_rows)

    print(df_srlt)
    print(df_g)

    srlt_fname = save_dir + 'sdp_rlt.csv'
    g_fname = save_dir + 'global.csv'

    df_srlt.to_csv(srlt_fname, index=False)
    df_g.to_csv(g_fname, index=False)


def main():
    experiment()


if __name__ == '__main__':
    main()
