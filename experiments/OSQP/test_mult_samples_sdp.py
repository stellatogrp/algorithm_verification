import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as spa

from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
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


def OSQP_CP_noboxstack(n, m, N=1, eps_b=.1, b_sample=None, solver_type='SDP', add_RLT=True, verbose=False):
    #  r = 1

    In = spa.eye(n)
    Im = spa.eye(m)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A
    Phalf = np.random.randn(n, n)
    P = Phalf.T @ Phalf
    # print(A)

    # b_const = spa.csc_matrix(np.zeros((n, 1)))
    zeros_n = np.zeros((n, 1))
    zeros_m = np.zeros((m, 1))
    l = 2 * np.ones((m, 1))
    u = 4 * np.ones((m, 1))
    sigma = 1
    rho = 1
    rho_inv = 1 / rho

    x = Iterate(n, name='x')
    y = Iterate(m, name='y')
    w = Iterate(m, name='w')
    z_tilde = Iterate(m, name='z_tilde')
    z = Iterate(m, name='z')
    b = Parameter(n, name='b')

    # step 1
    s1_Dtemp = P + sigma * In + rho * ATA
    s1_Atemp = spa.bmat([[sigma * In, rho*A.T, -rho * A.T, -In]])
    s1_D = In
    s1_A = spa.csc_matrix(np.linalg.inv(s1_Dtemp) @ s1_Atemp)
    step1 = HighLevelLinearStep(x, [x, z, y, b], D=s1_D, A=s1_A, b=zeros_n, Dinv=s1_D)

    # step 2
    s2_D = Im
    s2_A = spa.bmat([[Im, rho * A, rho * Im]])
    step2 = HighLevelLinearStep(y, [y, x, z], D=s2_D, A=s2_A, b=zeros_m, Dinv=s2_D)

    # step 3
    s3_D = Im
    s3_A = spa.bmat([[A, 1/rho * Im]])
    step3 = HighLevelLinearStep(w, [x, y], D=s3_D, A=s3_A, b=zeros_m, Dinv=s3_D)

    # step 4
    step4 = MaxWithVecStep(z_tilde, w, l=l)

    # step 5
    step5 = MinWithVecStep(z, z_tilde, u=u)

    # step 6 for fixed point residual
    s = Iterate(m, name='s')
    s6_D = Im
    s6_A = spa.bmat([[Im, rho_inv * Im]])
    step6 = HighLevelLinearStep(s, [z, y], D=s6_D, A=s6_A, b=zeros_m, Dinv=s6_D)

    # steps = [step1, step2, step3, step4, step5]
    steps = [step1, step2, step3, step4, step5, step6]

    # xset = CenteredL2BallSet(x, r=r)
    # x_l = -.1 * np.ones((n, 1))
    # x_u = 0.1 * np.ones((n, 1))
    # xset = BoxSet(x, x_l, x_u)
    xset = ConstSet(x, np.zeros((n, 1)))

    yset = ConstSet(y, np.zeros((m, 1)))

    zset = ConstSet(z, np.zeros((m, 1)))

    b_l = - eps_b * np.ones((n, 1))
    b_u = eps_b * np.ones((n, 1))
    # b_l = b_sample - eps_b
    # b_u = b_sample + eps_b
    bset = BoxSet(b, b_l, b_u)
    # bset = BoxSet
    # bset = ConstSet(b, 0.5 * np.ones((n, 1)))

    # obj = [ConvergenceResidual(x), ConvergenceResidual(y), ConvergenceResidual(z)]
    obj = [ConvergenceResidual(x), ConvergenceResidual(s)]
    # obj = OuterProdTrace(x)

    CP = CertificationProblem(N, [xset, yset, zset], [bset], obj, steps)

    # CP.print_cp()

    # res = CP.solve(solver_type='SDP', add_RLT=False, verbose=True)
    # print('sdp', res)
    # res = CP.solve(solver_type='SDP', add_RLT=True, verbose=True)
    # print('sdp rlt', res)
    # resg = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600)
    # print('global', resg)
    if solver_type == 'SDP':
        # return CP.solve(solver_type='SDP', add_RLT=add_RLT, verbose=verbose)
        return CP.canonicalize(solver_type='SDP', add_RLT=add_RLT, verbose=verbose)
    if solver_type == 'GLOBAL':
        # return CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600, verbose=verbose)
        return CP.canonicalize(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600, verbose=verbose)


def get_b_sample(n, eps_b):
    return np.random.uniform(-eps_b, eps_b, n).reshape((n, 1))


def mult_sample_expectation(n, m, N=10, eps_b=.01, solver_type='SDP'):
    overall_obj = 0
    overall_constraints = []
    for i in range(N):
        b_sample = get_b_sample(n, eps_b)
        solver = OSQP_CP_noboxstack(n, m, N=2, eps_b=eps_b, b_sample=b_sample)
        # res = solver.solve()
        # print(res)
        overall_obj += solver.handler.sdp_obj
        overall_constraints += solver.handler.sdp_constraints
    prob = cp.Problem(cp.Maximize(overall_obj), overall_constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=True)
    print(res / N)
    time = prob.solver_stats.solve_time
    return res, time


def test_multiple_eps(n, m, save_dir, max_N=2, verbose=False):
    # epsb_vals = [.01, .02, .05, .1, .2, .5]
    epsb_vals = [.01, .02]
    N_vals = range(2, max_N+1)
    # N_vals = [4]
    # res_s_rows = []
    res_srlt_rows = []
    res_g_rows = []
    for N in N_vals:
        for eps in epsb_vals:
            print('N:', N, 'eps:', eps)
            # res_s = OSQP_CP_noboxstack(n, m, N=N, eps_b=eps, solver_type='SDP', add_RLT=False, verbose=verbose)
            res_srlt = OSQP_CP_noboxstack(n, m, N=N, eps_b=eps, solver_type='SDP', add_RLT=True, verbose=verbose)
            res_g = OSQP_CP_noboxstack(n, m, N=N, eps_b=eps, solver_type='GLOBAL', verbose=verbose)

            # res_s_row = pd.Series(
            #     {
            #         'num_iter': N,
            #         'eps_b': eps,
            #         'obj': res_s[0],
            #         'solve_time': res_s[1],
            #     }
            # )
            # res_s_rows.append(res_s_row)

            res_srlt_row = pd.Series(
                {
                    'num_iter': N,
                    'eps_b': eps,
                    'obj': res_srlt[0],
                    'solve_time': res_srlt[1],
                }
            )
            res_srlt_rows.append(res_srlt_row)

            res_g_row = pd.Series(
                {
                    'num_iter': N,
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
    exit(0)

    # s_fname = save_dir + 'sdp.csv'
    srlt_fname = save_dir + 'sdp_rlt.csv'
    g_fname = save_dir + 'global.csv'

    # df_s.to_csv(s_fname, index=False)
    df_srlt.to_csv(srlt_fname, index=False)
    df_g.to_csv(g_fname, index=False)


def main():
    # N = 10
    m = 5
    n = 3
    # max_N = 6
    # OSQP_CP_noboxstack(n, m, N=N)
    # solver_type = 'SDP'
    # solver_type = 'GLOBAL'
    # print(OSQP_CP_noboxstack(n, m, N=4, eps_b=.01, solver_type=solver_type, add_RLT=True, verbose=True))
    save_dir = \
        '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/OSQP/data/mult_eps_test/'
    test_multiple_eps(n, m, save_dir, max_N=2, verbose=True)
    # mult_sample_expectation(n, m, N=3)


if __name__ == '__main__':
    main()
