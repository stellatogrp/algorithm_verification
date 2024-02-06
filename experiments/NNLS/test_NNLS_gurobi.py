import numpy as np
import scipy.sparse as spa

from algoverify import VerificationProblem
from algoverify.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algoverify.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algoverify.init_set.centered_l2_ball_set import CenteredL2BallSet
from algoverify.init_set.ellipsoidal_set import EllipsoidalSet
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.variables.iterate import Iterate
from algoverify.variables.parameter import Parameter


def centered_l2_xset(*args):
    x = args[0]
    r = args[1]
    return CenteredL2BallSet(x, r=r)


def off_center_l2_xset(*args):
    x = args[0]
    r = args[1]
    c = args[2]
    n = x.get_dim()
    Q = (1 / r ** 2) * np.eye(n)
    return EllipsoidalSet(x, Q, c)


def NNLS_cert_prob(n, m, A, N=1, t=.05, xset=None, bset_func=None):
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
    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')
    # print(y, x, b)

    ATA = A.T @ A
    In = spa.eye(n)
    C = spa.bmat([[In - t * ATA, t * A.T]])
    zeros = np.zeros(n)
    r = 1

    # xset = CenteredL2BallSet(x, r=r)
    xset = off_center_l2_xset(x, r, np.zeros(n))
    bset = CenteredL2BallSet(b, r=r)

    step1 = HighLevelLinearStep(y, [x, b], D=In, A=C, b=zeros)
    step2 = NonNegProjStep(x, y)

    steps = [step1, step2]

    obj = ConvergenceResidual(x)
    CP = VerificationProblem(N, [xset], [bset], obj, steps)
    res = CP.solve(solver_type='GLOBAL')
    print(res)


def main():
    np.random.seed(0)
    m = 5
    n = 3
    N = 1
    A = np.random.randn(m, n)
    NNLS_cert_prob(n, m, spa.csc_matrix(A), N=N, t=.05)


if __name__ == '__main__':
    main()
