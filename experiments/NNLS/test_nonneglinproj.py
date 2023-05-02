import numpy as np
import scipy.sparse as spa

from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import \
    NonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.high_level_alg_steps.nonneg_lin_step import NonNegLinStep
from algocert.init_set.box_set import BoxSet
# from algocert.init_set.const_set import ConstSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


def NNLS_cert_prob(n, m, A, K=1, t=.05, solver_type='SDP'):
    ATA = A.T @ A
    In = spa.eye(n)
    zeros_n = np.zeros((n, 1))
    # zeros_m = np.zeros((m, 1))
    ones_m = np.ones((m, 1))
    ones_n = np.ones((n, 1))

    C = spa.bmat([[In - t * ATA, t * A.T]])
    b = zeros_n

    x = Iterate(n, name='x')
    q = Parameter(m, name='q')

    step1 = NonNegLinStep(x, [x, q], C=C, b=b)
    steps = [step1]

    # initsets = [BoxSet(x, zeros_n, zeros_n)]
    initsets = [BoxSet(x, ones_n, 2 * ones_n)]
    # initsets = [ConstSet(x, zeros_n)]
    # initsets = [ConstSet(x, np.ones((n, 1)))]
    # paramsets = [ConstSet(q, np.ones((m, 1)))]
    paramsets = [BoxSet(q, 0.5 * ones_m, ones_m)]

    obj = [ConvergenceResidual(x)]

    CP = CertificationProblem(K, initsets, paramsets, obj, steps, num_samples=1)
    res = CP.solve(solver_type=solver_type, add_bounds=True,
                   add_RLT=False, TimeLimit=3600, minimize=False)
    # print('global', resg)
    return res


def NNLS_cert_prob_two_step(n, m, A, K=1, t=.05, solver_type='SDP'):
    ATA = A.T @ A
    In = spa.eye(n)
    zeros_n = np.zeros((n, 1))
    zeros_m = np.zeros((m, 1))
    ones_m = np.ones((m, 1))

    D = spa.eye(n, n)
    C = spa.bmat([[In - t * ATA, t * A.T]])
    b_const = zeros_n

    x = Iterate(n, name='x')
    y = Iterate(n, name='y')
    q = Parameter(m, name='q')

    step1 = HighLevelLinearStep(y, [x, q], D=D, A=C, b=b_const, Dinv=D)
    step2 = NonNegProjStep(x, y)
    # step2 = MaxWithVecStep(x, y, l=zeros_m)

    steps = [step1, step2]

    # initsets = [ConstSet(x, zeros_n)]
    initsets = [BoxSet(x, zeros_n, zeros_n)]
    # initsets = [ConstSet(x, np.ones((n, 1)))]
    # paramsets = [ConstSet(q, np.ones((m, 1)))]
    paramsets = [BoxSet(q, zeros_m, ones_m)]

    obj = [ConvergenceResidual(x)]

    CP = CertificationProblem(K, initsets, paramsets, obj, steps)
    resg = CP.solve(solver_type=solver_type, add_bounds=True, add_RLT=False,
                    TimeLimit=3600, minimize=False, verbose=True)
    # print('global', resg)
    return resg


def main():
    np.random.seed(0)
    m = 5
    n = 3
    K = 1
    A = np.random.randn(m, n)
    # s = 'GLOBAL'
    # s = 'SDP'
    # s = 'SDP_SCGAL'
    s = 'SDP_CGAL'
    res1 = NNLS_cert_prob(n, m, spa.csc_matrix(A), K=K, t=.05, solver_type=s)
    # res2 = NNLS_cert_prob_two_step(n, m, spa.csc_matrix(A), K=K, t=.05, solver_type='SDP')
    print('combined nonneglinstep:', res1)
    # print('split nonneglinstep:', res2)


if __name__ == '__main__':
    main()
