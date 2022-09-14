import numpy as np
import scipy.sparse as spa
import matplotlib.pyplot as plt

from certification_problem.certification_problem import CertificationProblem
from certification_problem.variables.iterate import Iterate
from certification_problem.variables.parameter import Parameter
from certification_problem.basic_algorithm_steps.block_step import BlockStep
from certification_problem.basic_algorithm_steps.linear_step import LinearStep
from certification_problem.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from certification_problem.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from certification_problem.basic_algorithm_steps.min_with_vec_step import MinWithVecStep

from certification_problem.high_level_alg_steps.hl_linear_step import HighLevelLinearStep

from certification_problem.init_set.box_set import BoxSet
from certification_problem.init_set.const_set import ConstSet
from certification_problem.init_set.centered_l2_ball_set import CenteredL2BallSet
from certification_problem.init_set.ellipsoidal_set import EllipsoidalSet
from certification_problem.init_set.linf_ball_set import LInfBallSet
from certification_problem.objectives.convergence_residual import ConvergenceResidual
from certification_problem.objectives.outer_prod_trace import OuterProdTrace


def test_OSQP_GLOBAL(N=1):
    print('--GLOBAL--')
    m = 5
    n = 3
    r = 1

    In = spa.eye(n)
    Im = spa.eye(m)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A
    Phalf = np.random.randn(n, n)
    P = Phalf.T @ Phalf
    # print(A)

    rho = 1
    sigma = 1

    # b_const = spa.csc_matrix(np.zeros((n, 1)))
    zeros_n = np.zeros((n, 1))
    zeros_m = np.zeros((m, 1))
    l = 2 * np.ones((m, 1))
    u = 4 * np.ones((m, 1))
    sigma = 1
    rho = 1

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

    steps = [step1, step2, step3, step4, step5]

    # xset = CenteredL2BallSet(x, r=r)
    x_l = -1 * np.ones((n, 1))
    x_u = np.ones((n, 1))
    xset = BoxSet(x, x_l, x_u)

    yset = ConstSet(y, np.zeros((m, 1)))

    zset = ConstSet(z, np.zeros((m, 1)))

    # bset = CenteredL2BallSet(b, r=r)
    b_l = np.ones((n, 1))
    b_u = 3 * np.ones((n, 1))
    bset = BoxSet(b, b_l, b_u)

    obj = ConvergenceResidual(x)
    # obj = OuterProdTrace(x)

    CP = CertificationProblem(N, [xset, yset, zset], [bset], obj, steps)

    # CP.print_cp()
    res = CP.solve(solver_type='GLOBAL', add_bounds=True)
    # res = CP.solve(solver_type='SDP', add_RLT=True)
    return res


def main():
    N = 1
    res_global = test_OSQP_GLOBAL(N=N)


if __name__ == '__main__':
    main()
