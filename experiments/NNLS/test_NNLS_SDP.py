#  import certification_problem.init_set as cpi
import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as spa

# from algocert.basic_algorithm_steps.block_step import BlockStep
# from algocert.basic_algorithm_steps.linear_step import LinearStep
from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.linear_step import LinearStep
from algocert.init_set.box_set import BoxSet

# from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter

#  from joblib import Parallel, delayed


def NNLS_cert_prob(n, m, A, K=1, t=.05, xset=None, bset=None, glob_include=True):
    '''
        Set up and solve certification problem for:
        min (1/2) || Ax - b ||_2^2 s.t. x >= 0
        via proximal gradient descent
    :param n: dimension of x
    :param m: dimension of b
    :param A:
    :param t: step size
    :return:
    '''
    ATA = A.T @ A
    In = spa.eye(n)
    # r = 1
    # x_l = np.zeros((n, 1))
    # x_u = np.ones((n, 1))
    # b_l = np.zeros((m, 1))
    # b_u = np.ones((m, 1))

    C = spa.bmat([[In - t * ATA, t * A.T]])
    D = spa.eye(n, n)
    b_const = spa.csc_matrix(np.zeros((n, 1)))
    b_const = np.zeros((n, 1))

    # u = Iterate(n + m, name='u')
    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    # step1 = BlockStep(u, [x, b])
    # step2 = LinearStep(y, u, A=C, D=D, b=b_const)
    step1 = LinearStep(y, [x, b], D=D, A=C, b=b_const, Dinv=D)
    step2 = NonNegProjStep(x, y)

    steps = [step1, step2]

    # xset = CenteredL2BallSet(x, r=r)
    # x_l = 10 + np.zeros((n, 1))
    # x_u = 10 + np.ones((n, 1))

    # x_l = np.zeros((n, 1))
    # x_u = np.zeros((n, 1))

    x_l = 0.5 * np.ones((n, 1))
    x_u = np.ones((n, 1))
    # x_u[0] = -1
    xset = BoxSet(x, x_l, x_u)

    # bset = CenteredL2BallSet(b, r=r)
    # b_l = np.zeros((m, 1))
    # b_u = np.ones((m, 1))
    b_l = 20 + np.zeros((m, 1))
    b_u = 20 + np.ones((m, 1))
    bset = BoxSet(b, b_l, b_u)

    obj = ConvergenceResidual(x)
    # CP = CertificationProblem(K, [xset], [bset], obj, steps)
    # CP2 = CertificationProblem(K, [xset], [bset], obj, steps)
    # CP3 = CertificationProblem(K, [xset], [bset], obj, steps)
    # CP4 = CertificationProblem(K, [xset], [bset], obj, steps)

    # # CP.problem_data = qp_problem_data
    # # CP.print_cp()
    # # res = CP.solve(solver_type='SDP_ADMM')
    # res = CP.solve(solver_type='SDP', add_RLT=False, add_planet=False)
    # res_r = CP2.solve(solver_type='SDP', add_RLT=True, add_planet=False)
    # res_p = CP3.solve(solver_type='SDP', add_RLT=True, add_planet=True)
    # res_g = CP4.solve(solver_type='GLOBAL')

    # print(res)
    # print(res_r)
    # print(res_p)
    # print(res_g)

    # out_fname = 'data/planet_test.csv'
    out = []
    # K = 2
    for K_curr in range(1, K+1):
        # K_curr = 1
        CP = CertificationProblem(K_curr, [xset], [bset], obj, steps)
        CP2 = CertificationProblem(K_curr, [xset], [bset], obj, steps)
        CP3 = CertificationProblem(K_curr, [xset], [bset], obj, steps)
        CP4 = CertificationProblem(K_curr, [xset], [bset], obj, steps)
        CP5 = CertificationProblem(K_curr, [xset], [bset], obj, steps)

        (sdp, sdptime) = CP.solve(solver_type='SDP', add_RLT=False, add_planet=False)
        (sdp_r, sdp_rtime) = CP2.solve(solver_type='SDP', add_RLT=True, add_planet=False)
        (sdp_p, sdp_ptime) = CP3.solve(solver_type='SDP', add_RLT=True, add_planet=True)
        if glob_include:
            (glob, glob_time) = CP4.solve(solver_type='GLOBAL', add_bounds=True)
        else:
            glob, glob_time = 0, 0
        (sdp_c, sdp_ctime) = CP5.solve(solver_type='SDP_CUSTOM')

        # print('sdp:', sdp)
        # exit(0)

        out.append(
            pd.Series({
                'K': K_curr,
                'sdp': sdp,
                'sdptime': sdptime,
                'sdp_r': sdp_r,
                'sdp_rtime': sdp_rtime,
                'sdp_p': sdp_p,
                'sdp_ptime': sdp_ptime,
                'sdp_c': sdp_c,
                'sdp_ctime': sdp_ctime,
                'glob': glob,
                'glob_time': glob_time,
            })
        )
    out_df = pd.DataFrame(out)
    print(out_df)
    # out_df.to_csv(out_fname, index=False)


def cp_test(A):
    m, n = A.shape
    b = 20 * np.ones(m)

    x = cp.Variable(n)
    obj = .5 * cp.sum_squares(A @ x - b)
    constraints = [x >= 0]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    res = prob.solve()
    print(res)
    print(np.round(x.value, 4))

    exit(0)


def main():
    np.random.seed(1)
    m = 5
    n = 3
    K = 3
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    # cp_test(A)
    NNLS_cert_prob(n, m, A, K=K, t=.05, glob_include=True)


if __name__ == '__main__':
    main()
