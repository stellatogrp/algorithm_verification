import numpy as np
import scipy.sparse as spa

from algocert.basic_algorithm_steps.block_step import BlockStep
from algocert.basic_algorithm_steps.linear_step import LinearStep
from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


def main():
    N = 1
    m = 4
    n = 2
    k = 3
    r = 1

    In = spa.eye(n)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A
    # print(A)

    t = .05

    C = spa.bmat([[In - t * ATA, t * A.T]])

    #  b_l = 1
    #  b_u = 3

    u = Iterate(n + n + k, name='u')
    #  w = Iterate(n + m + k, name='w')
    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(k, name='b')

    # step1 = BlockStep(u, [x, b])
    # step2 = LinearStep(y, C, u)
    # step3 = NonNegProjStep(x, y)
    # steps = [step1, step2, step3]
    # print(step1.get_output_var().name, step2.get_output_var().name, step3.get_output_var().name)
    step1 = BlockStep(u, [x, y, b])

    C = np.ones((n, n + n + k))
    step2 = LinearStep(y, C, u)
    step3 = NonNegProjStep(x, y)

    # steps = [step1, step2, step3, step4]
    steps = [step3, step2, step1]

    xset = CenteredL2BallSet(x, np.zeros(n), r=r)
    # bset = BoxSet(b, b_l, b_u)
    bset = CenteredL2BallSet(b, np.zeros(k), r=r)

    obj = ConvergenceResidual(x)

    CP = CertificationProblem(N, [xset], [bset], obj, steps)
    # CP.print_cp()
    CP.solve()


if __name__ == '__main__':
    main()
