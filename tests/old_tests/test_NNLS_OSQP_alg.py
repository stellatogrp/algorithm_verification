import numpy as np

from algoverify.basic_algorithm_steps.block_step import BlockStep
from algoverify.basic_algorithm_steps.linear_step import LinearStep
from algoverify.init_set.centered_l2_ball_set import CenteredL2BallSet
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.variables.iterate import Iterate
from algoverify.variables.parameter import Parameter
from algoverify.verification_problem import VerificationProblem


def main():
    N = 1
    m = 5
    n = 3
    r = 1

    #  In = spa.eye(n)

    np.random.seed(0)
    P = np.random.randn(m, n)
    #  P = spa.csc_matrix(A)
    # print(A)

    #  rho = .05
    #  sigma = 1
    #
    #  In = spa.eye(n, n)
    #  Im = spa.eye(m, m)
    #  zeros_n = np.zeros((n, 1))
    #  zeros_m = np.zeros((m, 1))

    x = Iterate(n, name='x')
    y = Iterate(m, name='y')
    z = Iterate(m, name='z')
    q = Parameter(m, name='b')

    xset = CenteredL2BallSet(x, np.zeros(n), r=r)
    yset = CenteredL2BallSet(y, np.zeros(m), r=.01)
    zset = CenteredL2BallSet(z, np.zeros(m), r=.01)
    # bset = BoxSet(b, b_l, b_u)
    qset = CenteredL2BallSet(q, np.zeros(m), r=r)

    xblock = Iterate(n + 3 * m, name='xblock')
    step1 = BlockStep(xblock, [x, y, z, q])
    #  xblock_mat = spa.bmat([[]])
    step2 = LinearStep(x, xblock, P)

    steps = [step1, step2]
    obj = ConvergenceResidual(x)

    CP = VerificationProblem(N, [xset, yset, zset], [qset], obj, steps)
    # CP.print_cp()
    CP.solve()


if __name__ == '__main__':
    main()
