#  import certification_problem.init_set as cpi
import numpy as np
import scipy.sparse as spa
from certification_problem.basic_algorithm_steps.block_step import BlockStep
from certification_problem.basic_algorithm_steps.linear_step import LinearStep
from certification_problem.basic_algorithm_steps.nonneg_orthant_proj_step import \
    NonNegProjStep
from certification_problem.certification_problem import CertificationProblem
from certification_problem.init_set.centered_l2_ball_set import \
    CenteredL2BallSet
from certification_problem.objectives.convergence_residual import \
    ConvergenceResidual
from certification_problem.variables.iterate import Iterate
from certification_problem.variables.parameter import Parameter

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

    C = spa.bmat([[In - t * ATA, t * A.T]])
    D = spa.eye(n, n)
    b_const = spa.csc_matrix(np.zeros((n, 1)))

    u = Iterate(n + m, name='u')
    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    step1 = BlockStep(u, [x, b])
    step2 = LinearStep(y, u, A=C, D=D, b=b_const)
    step3 = NonNegProjStep(x, y)

    steps = [step1, step2, step3]

    xset = CenteredL2BallSet(x, r=r)

    bset = CenteredL2BallSet(b, r=r)

    obj = ConvergenceResidual(x)
    CP = CertificationProblem(N, [xset], [bset], obj, steps)

    # CP.problem_data = qp_problem_data
    # CP.print_cp()
    CP.solve(solver_type='SDP')
    # print(res)


def main():
    np.random.seed(0)
    m = 25
    n = 10
    N = 2
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    NNLS_cert_prob(n, m, A, N=N, t=.05)


if __name__ == '__main__':
    main()
