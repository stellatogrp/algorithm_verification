# import cvxpy as cp
import numpy as np
# import pandas as pd
import scipy.sparse as spa

# from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
# from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import \
#     NonNegProjStep
from algocert.basic_algorithm_steps.partial_nonneg_orthant_proj_step import \
    PartialNonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.init_set.box_set import BoxSet
# from algocert.init_set.box_stack_set import BoxStackSet
# from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
# from algocert.init_set.const_set import ConstSet
# from algocert.init_set.control_example_set import ControlExampleSet
# from algocert.init_set.init_set import InitSet
from algocert.objectives.convergence_residual import ConvergenceResidual
# from algocert.objectives.l1_conv_resid import L1ConvResid
# from algocert.utils.plotter import plot_results
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


def NUM_experiment(m_orig, n, K=1):
    R = np.random.binomial(n=2, p=0.25, size=(m_orig, n))
    print(R)

    A, M = form_NUM_matrices(R)
    m, n = A.shape
    k = m + n
    Ik = spa.eye(k)

    # self.w_sample = np.random.uniform(0, 1, size=self.n)
    # self.t_sample = 1 * np.ones(self.n)

    w = np.random.uniform(0, 1, size=n)
    t = np.ones(n)
    c_l = -np.ones(m_orig)  # np.zeros(m_orig)
    c_u = np.ones(m_orig)

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
    s1_D = Ik
    s1_Atemp = spa.bmat([[Ik, -Ik]])
    s1_A = MpIinv @ s1_Atemp
    s1_b = np.zeros((k, 1))

    step1 = HighLevelLinearStep(u, [z, q], D=s1_D, A=s1_A, b=s1_b, Dinv=s1_D)

    # step 2
    s2_D = Ik
    s2_A = spa.bmat([[2 * Ik, -Ik]])
    s2_b = np.zeros((k, 1))

    step2 = HighLevelLinearStep(w, [u, z], D=s2_D, A=s2_A, b=s2_b, Dinv=s2_D)

    # step 3
    # step3 = NonNegProjStep(u_tilde, w)
    nonneg_ranges = (n, m + n)
    step3 = PartialNonNegProjStep(u_tilde, w, nonneg_ranges)

    # step 4
    s4_D = Ik
    s4_A = spa.bmat([[Ik, Ik, -Ik]])
    s4_b = np.zeros((k, 1))

    step4 = HighLevelLinearStep(z, [z, u_tilde, u], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)

    steps = [step1, step2, step3, step4]

    # for the iterate/parameter sets
    qset = BoxSet(q, q_l, q_u)
    zset = BoxSet(z, np.zeros((k, 1)), np.zeros((k, 1)))

    obj = [ConvergenceResidual(z)]

    CP = CertificationProblem(K, [zset], [qset], obj, steps)

    resg = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600)
    print('global', resg)


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
    np.random.seed(0)
    m = 2
    n = 3
    K = 1
    NUM_experiment(m, n, K=K)


if __name__ == '__main__':
    main()
