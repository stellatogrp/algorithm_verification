from joblib import Parallel, delayed
import numpy as np
import scipy.sparse as spa
from certification_problem.certification_problem import CertificationProblem
from certification_problem.variables.iterate import Iterate
from certification_problem.variables.parameter import Parameter
from certification_problem.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from certification_problem.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
import certification_problem.init_set as cpi
from certification_problem.init_set.centered_l2_ball_set import CenteredL2BallSet
from certification_problem.init_set.ellipsoidal_set import EllipsoidalSet
from certification_problem.objectives.convergence_residual import ConvergenceResidual


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
    CP = CertificationProblem(N, [xset], [bset], obj, steps)
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
