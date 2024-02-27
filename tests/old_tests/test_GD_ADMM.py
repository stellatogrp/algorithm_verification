#  import verification_problem.init_set as cpi
import numpy as np
import scipy.sparse as spa

from algoverify.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algoverify.init_set.centered_l2_ball_set import CenteredL2BallSet
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.variables.iterate import Iterate
from algoverify.variables.parameter import Parameter
from algoverify.verification_problem import VerificationProblem

# from algoverify.basic_algorithm_steps.block_step import BlockStep
# from algoverify.basic_algorithm_steps.linear_step import LinearStep
# from algoverify.basic_algorithm_steps.nonneg_orthant_proj_step import \
# NonNegProjStep
# from algoverify.init_set.box_set import BoxSet

#  from joblib import Parallel, delayed


def NNLS_cert_prob(n, m, A, N=1, t=.05, xset=None, bset=None):
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
    r = 1
    # x_l = np.zeros((n, 1))
    # x_u = np.ones((n, 1))
    # b_l = np.zeros((m, 1))
    # b_u = np.ones((m, 1))

    C = spa.bmat([[In - t * ATA, t * A.T]])
    D = spa.eye(n, n)
    b_const = np.zeros((n, 1))

    # u = Iterate(n + m, name='u')
    # y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    # step1 = BlockStep(u, [x, b])
    # step2 = LinearStep(y, u, A=C, D=D, b=b_const)
    step1 = HighLevelLinearStep(x, [x, b], D=D, A=C, b=b_const, Dinv=D)
    # step2 = NonNegProjStep(x, y)

    steps = [step1]

    xset = CenteredL2BallSet(x, r=r)
    # xset = BoxSet(x, x_l, x_u)

    bset = CenteredL2BallSet(b, r=r)
    # bset = BoxSet(b, b_l, b_u)

    obj = ConvergenceResidual(x)
    CP = VerificationProblem(N, [xset], [bset], obj, steps)

    # CP.problem_data = qp_problem_data
    # CP.print_cp()
    # res = CP.solve(solver_type='SDP_ADMM')
    res = CP.solve(solver_type='SDP')
    print(res)


def main():
    np.random.seed(0)
    m = 5
    n = 3
    N = 1
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    NNLS_cert_prob(n, m, A, N=N, t=.05)


if __name__ == '__main__':
    main()
