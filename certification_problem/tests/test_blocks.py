import numpy as np
import scipy.sparse as spa

from certification_problem.certification_problem import CertificationProblem
from certification_problem.variables.iterate import Iterate
from certification_problem.variables.parameter import Parameter
from certification_problem.algorithm_steps.block_step import BlockStep
from certification_problem.algorithm_steps.linear_step import LinearStep
from certification_problem.algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from certification_problem.init_set.box_set import BoxSet
from certification_problem.init_set.l2_ball import L2BallSet
from certification_problem.objectives.convergence_residual import ConvergenceResidual


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

    b_l = 1
    b_u = 3

    u = Iterate(n + n + k, name='u')
    w = Iterate(n + m + k, name='w')
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

    xset = L2BallSet(x, np.zeros(n), r=r)
    # bset = BoxSet(b, b_l, b_u)
    bset = L2BallSet(b, np.zeros(k), r=r)

    obj = ConvergenceResidual(x)

    CP = CertificationProblem(N, [xset], [bset], obj, steps)
    # CP.print_cp()
    CP.solve()


if __name__ == '__main__':
    main()
