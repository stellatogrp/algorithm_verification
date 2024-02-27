import numpy as np
import pandas as pd
import scipy.sparse as spa

# from algoverify.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algoverify.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algoverify.high_level_alg_steps.linear_step import LinearStep
from algoverify.init_set.box_set import BoxSet
from algoverify.init_set.zero_set import ZeroSet

# from algoverify.init_set.box_stack_set import BoxStackSet
# from algoverify.init_set.centered_l2_ball_set import CenteredL2BallSet
# from algoverify.init_set.const_set import ConstSet
# from algoverify.init_set.control_example_set import ControlExampleSet
# from algoverify.init_set.init_set import InitSet
from algoverify.objectives.convergence_residual import ConvergenceResidual

# from algoverify.objectives.l1_conv_resid import L1ConvResid
# from algoverify.utils.plotter import plot_results
from algoverify.variables.iterate import Iterate
from algoverify.variables.parameter import Parameter
from algoverify.verification_problem import VerificationProblem


def NUM_single(m_orig, n, K=1, sdp_include=False, glob_include=True):
    R = np.random.binomial(n=1, p=0.25, size=(m_orig, n))
    print(R)

    A, M = form_NUM_matrices(R)
    # print(A)
    # exit(0)
    m, n = A.shape
    k = m + n
    Ik = spa.eye(k)

    # self.w_sample = np.random.uniform(0, 1, size=self.n)
    # self.t_sample = 1 * np.ones(self.n)

    w = -np.random.uniform(0, 1, size=n)
    t = 1 * np.ones(n)
    c_l = .9 * np.ones(m_orig)
    c_u = 1 * np.ones(m_orig)

    # q_l = np.hstack([w - 1, c_l, np.zeros(n) - .1, t])
    q_l = np.hstack([w, c_l, np.zeros(n), t])
    q_u = np.hstack([w, c_u, np.zeros(n), t])
    print(q_l.shape)

    q_l = q_l.reshape(-1, 1)
    q_u = q_u.reshape(-1, 1)
    # print(q_l)
    # print(q_u)

    MpI = spa.csc_matrix(M) + spa.eye(m + n)
    # print(MpI.toarray())
    MpIinv = spa.linalg.inv(MpI)

    z = Iterate(k, name='z')
    u = Iterate(k, name='u')
    w = Iterate(k, name='w')
    u_tilde = Iterate(k, name='u_tilde')
    q = Parameter(k, name='q')
    qset = BoxSet(q, q_l, q_u)

    # step 1
    # s1_D = Ik
    # s1_Atemp = spa.bmat([[Ik, -Ik]])
    # s1_A = MpIinv @ s1_Atemp
    # s1_b = np.zeros((k, 1))

    # step1 = LinearStep(u, [z, q], D=s1_D, A=s1_A, b=s1_b, Dinv=s1_D)

    s1_D = MpI
    s1_A = spa.bmat([[Ik, -Ik]])
    s1_b = np.zeros((k, 1))

    step1 = LinearStep(u, [z, q], D=s1_D, A=s1_A, b=s1_b, Dinv=MpIinv)

    # step 2
    s2_D = Ik
    s2_A = spa.bmat([[2 * Ik, -Ik]])
    s2_b = np.zeros((k, 1))

    step2 = LinearStep(w, [u, z], D=s2_D, A=s2_A, b=s2_b, Dinv=s2_D)

    # step 3
    nonneg_ranges = (n, m + n)
    # nonneg_ranges = (m + n - 1, m + n)
    # step3 = PartialNonNegProjStep(u_tilde, w, nonneg_ranges)
    step3 = NonNegProjStep(u_tilde, w, nonneg_ranges=nonneg_ranges)
    # step3 = NonNegProjStep(u_tilde, w, nonneg_ranges=None)
    # step3 = LinearStep(u_tilde, [w], D=Ik, A=Ik, b=s2_b, Dinv=Ik)

    # step 4
    s4_D = Ik
    s4_A = spa.bmat([[Ik, Ik, -Ik]])
    s4_b = np.zeros((k, 1))

    step4 = LinearStep(z, [z, u_tilde, u], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)
    # step4 = LinearStep(z, [z, w, u], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)

    steps = [step1, step2, step3, step4]
    # steps = [step1, step2, step4]

    # for the iterate/parameter sets
    qset = BoxSet(q, q_l, q_u)
    # zset = BoxSet(z, np.zeros((k, 1)), np.zeros((k, 1)))
    zset = ZeroSet(z)

    # zset = BoxSet(z, -np.ones((k, 1)), np.ones((k, 1)))

    obj = [ConvergenceResidual(z)]
    # obj = [ConvergenceResidual(u_tilde)]

    VerificationProblem(K, [zset], [qset], obj, steps)
    VerificationProblem(K, [zset], [qset], obj, steps)

    out = []
    # K = 2
    for K_curr in range(1, K+1):
        # K_curr = 2
        # CP = VerificationProblem(K_curr, [zset], [qset], obj, steps)
        # CP2 = VerificationProblem(K_curr, [zset], [qset], obj, steps)
        # CP3 = VerificationProblem(K_curr, [zset], [qset], obj, steps)
        CP4 = VerificationProblem(K_curr, [zset], [qset], obj, steps)

        CP5 = VerificationProblem(K_curr, [zset], [qset], obj, steps)
        if sdp_include:
            out_sdp = CP5.solve(solver_type='SDP_CUSTOM')
            sdp_c, sdp_ctime = out_sdp['sdp_objval'], out_sdp['sdp_solvetime']
        else:
            sdp_c, sdp_ctime = 0, 0
        # sdp_c, sdp_ctime = 0, 0
        # print(sdp_c, sdp_ctime)
        # exit(0)

        # (sdp, sdptime) = CP.solve(solver_type='SDP', add_RLT=False, add_planet=False)
        # (sdp_r, sdp_rtime) = CP2.solve(solver_type='SDP', add_RLT=True, add_planet=False)
        # (sdp_p, sdp_ptime) = CP3.solve(solver_type='SDP', add_RLT=True, add_planet=True)
        if glob_include:
            (glob, glob_time) = CP4.solve(solver_type='GLOBAL', add_bounds=True)
        else:
            glob, glob_time = 0, 0


        out.append(
            pd.Series({
                'K': K_curr,
                # 'sdp': sdp,
                # 'sdptime': sdptime,
                # 'sdp_r': sdp_r,
                # 'sdp_rtime': sdp_rtime,
                # 'sdp_p': sdp_p,
                # 'sdp_ptime': sdp_ptime,
                'sdp_c': sdp_c,
                'sdp_ctime': sdp_ctime,
                'glob': glob,
                'glob_time': glob_time,
            })
        )
        # print(out)
        # exit(0)
    out_df = pd.DataFrame(out)
    print(out_df)
    # out_df.to_csv('experiments/NUM/data/test_custom_cross.csv', index=False)

    # res = CP.solve(solver_type='SDP', add_RLT=True, add_planet=True)

    # resg = CP2.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600)
    # print('sdp', res)
    # print('global', resg)


def form_NUM_matrices(R):
    (m_orig, n) = R.shape
    A = np.vstack([R, -np.eye(n), np.eye(n)])
    # print(A)

    M = spa.bmat([
        [None, A.T],
        [-A, None]
    ]).toarray()
    # print(M)

    print('A.shape:', A.shape)
    print('M.shape:', M.shape)

    return A, M


def main():
    np.random.seed(1)
    m = 1
    n = 2
    K = 2
    NUM_single(m, n, K=K, sdp_include=True, glob_include=True)


if __name__ == '__main__':
    main()
