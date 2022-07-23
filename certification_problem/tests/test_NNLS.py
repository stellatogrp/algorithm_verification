import numpy as np
import scipy.sparse as spa

from certification_problem.certification_problem import CertificationProblem
from certification_problem.variables.iterate import Iterate
from certification_problem.variables.parameter import Parameter
from certification_problem.basic_algorithm_steps.block_step import BlockStep
from certification_problem.basic_algorithm_steps.linear_step import LinearStep
from certification_problem.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from certification_problem.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep

from certification_problem.high_level_alg_steps.hl_linear_step import HighLevelLinearStep

from certification_problem.init_set.box_set import BoxSet
from certification_problem.init_set.centered_l2_ball_set import CenteredL2BallSet
from certification_problem.init_set.ellipsoidal_set import EllipsoidalSet
from certification_problem.init_set.linf_ball_set import LInfBallSet
from certification_problem.objectives.convergence_residual import ConvergenceResidual


def test_NNLS_SDP():
    print('----SDP----')
    N = 1
    m = 5
    n = 3
    r = 1

    In = spa.eye(n)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A
    # print(A)

    t = .05

    C = spa.bmat([[In - t * ATA, t * A.T]])
    D = spa.eye(n, n)
    b_const = spa.csc_matrix(np.zeros((n, 1)))

    b_l = 1
    b_u = 3

    u = Iterate(n + m, name='u')
    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    step1 = BlockStep(u, [x, b])
    step2 = LinearStep(y, u, A=C, D=D, b=b_const)
    step3 = NonNegProjStep(x, y)
    zeros = np.zeros((n, 1))
    ones = np.ones((n, 1))
    # step3 = MaxWithVecStep(x, y, l=ones)
    steps = [step1, step2, step3]
    # print(step1.get_output_var().name, step2.get_output_var().name, step3.get_output_var().name)

    Q = np.eye(n)
    c = np.zeros((n, 1))
    xset = CenteredL2BallSet(x, r=r)
    # xset = EllipsoidalSet(x, Q=Q, c=c)

    bset = CenteredL2BallSet(b, r=r)
    # bset = BoxSet(b, b_l, b_u)
    # c = 2 * np.ones((m, 1))
    # r = 1
    # bset = LInfBallSet(b, c, r)

    obj = ConvergenceResidual(x)
    qp_problem_data = {'A': .5 * ATA}
    CP = CertificationProblem(N, [xset], [bset], obj, steps, qp_problem_data=qp_problem_data)

    # CP.problem_data = qp_problem_data
    # CP.print_cp()
    CP.solve(solver_type='SDP')


def test_NNLS_GLOBAL():
    print('--GLOBAL--')
    N = 1
    m = 5
    n = 3
    r = 1

    In = spa.eye(n)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A
    # print(A)

    t = .05

    C = spa.bmat([[In - t * ATA, t * A.T]])
    D = spa.eye(n, n)
    # b_const = spa.csc_matrix(np.zeros((n, 1)))
    b_const = spa.csc_matrix(np.zeros(n))

    b_l = 1
    b_u = 3

    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    step1 = HighLevelLinearStep(y, [x, b], D=D, A=C, b=b_const)
    step2 = NonNegProjStep(x, y)
    # step2 = HighLevelLinearStep(x, [y], D=D, A=D, b=b_const)

    steps = [step1]
    steps = [step1, step2]

    xset = CenteredL2BallSet(x, r=r)
    # xset = EllipsoidalSet(x, Q=Q, c=c)

    bset = CenteredL2BallSet(b, r=r)
    # # bset = BoxSet(b, b_l, b_u)
    # c = 2 * np.ones((m, 1))
    # r = 1
    # bset = LInfBallSet(b, c, r)

    obj = ConvergenceResidual(x)

    CP = CertificationProblem(N, [xset], [bset], obj, steps)

    # CP.print_cp()
    CP.solve(solver_type='GLOBAL')


def main():
    test_NNLS_SDP()
    # test_NNLS_GLOBAL()


if __name__ == '__main__':
    main()
