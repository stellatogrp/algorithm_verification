import numpy as np
import scipy.sparse as spa

from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.linear_step import LinearStep
from algocert.init_set.box_set import BoxSet
from algocert.init_set.box_stack_set import BoxStackSet
from algocert.init_set.const_set import ConstSet
from algocert.init_set.control_example_set import ControlExampleSet
from algocert.objectives.convergence_residual import ConvergenceResidual

# from algocert.init_set.init_set import InitSet
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


def OSQP_cert_prob(n, m, K=1, t=.05, xset=None, bset_func=None):
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
    step1 = LinearStep(x, [x, z, y, b], D=s1_D, A=s1_A, b=zeros_n, Dinv=s1_D)

    # step 2
    s2_D = Im
    s2_A = spa.bmat([[Im, rho * A, rho * Im]])
    step2 = LinearStep(y, [y, x, z], D=s2_D, A=s2_A, b=zeros_m, Dinv=s2_D)

    # step 3
    s3_D = Im
    s3_A = spa.bmat([[A, 1/rho * Im]])
    step3 = LinearStep(w, [x, y], D=s3_D, A=s3_A, b=zeros_m, Dinv=s3_D)

    # step 4
    step4 = MaxWithVecStep(z_tilde, w, l=l)

    # step 5
    step5 = MinWithVecStep(z, z_tilde, u=u)

    # step 6 for fixed point residual
    s = Iterate(m, name='s')
    s6_D = Im
    s6_A = spa.bmat([[Im, rho_inv * Im]])
    step6 = LinearStep(s, [z, y], D=s6_D, A=s6_A, b=zeros_m, Dinv=s6_D)

    # steps = [step1, step2, step3, step4, step5]
    steps = [step1, step2, step3, step4, step5, step6]

    # xset = CenteredL2BallSet(x, r=r)
    x_l = -1 * np.ones((n, 1))
    x_u = np.ones((n, 1))
    xset = BoxSet(x, x_l, x_u)
    xset = ConstSet(x, np.zeros((n, 1)))

    yset = ConstSet(y, np.zeros((m, 1)))

    zset = ConstSet(z, np.zeros((m, 1)))

    # bset = CenteredL2BallSet(b, r=r)
    offset = 3
    b_l = np.ones((n-offset, 1))
    b_u = 10 * np.ones((n-offset, 1))
    # bset = BoxSet(b, b_l, b_u)
    bset = ControlExampleSet(b, b_l, b_u)
    # bset = ConstSet(b, np.zeros((n, 1)))

    test = Parameter(n-offset, name='test')
    test_l = np.ones((n-offset, 1))
    test_u = 10 * np.ones((n-offset, 1))
    offset_zeros = np.zeros((offset, 1))
    test_set = BoxSet(test, test_l, test_u)
    bset = BoxStackSet(b, [test_set, [offset_zeros, offset_zeros]])

    # obj = [ConvergenceResidual(x), ConvergenceResidual(y), ConvergenceResidual(z)]
    obj = [ConvergenceResidual(x), ConvergenceResidual(s)]
    # obj = OuterProdTrace(x)

    CP = CertificationProblem(K, [xset, yset, zset], [test_set, bset], obj, steps)

    CP.print_cp()

    # res = CP.solve(solver_type='SDP', add_RLT=False, verbose=True)
    # print('sdp', res)
    # res = CP.solve(solver_type='SDP', add_RLT=True, verbose=True)
    # print('sdp rlt', res)
    resg = CP.solve(solver_type='GLOBAL', add_bounds=False, TimeLimit=3600)
    print('global', resg)


def main():
    m = 5
    n = 3
    K = 1
    OSQP_cert_prob(n, m, K=K)
    # OSQP_CP_noboxstack(n, m, N=N)


if __name__ == '__main__':
    main()
