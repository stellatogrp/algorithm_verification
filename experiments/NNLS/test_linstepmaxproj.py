import numpy as np
import scipy.sparse as spa

from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.linear_max_proj_step import LinearMaxProjStep
from algocert.init_set.box_set import BoxSet
from algocert.init_set.l2_ball_set import L2BallSet

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

    l = 0.1 * np.ones((n, 1))
    step1 = LinearMaxProjStep(x, [x, q], A=C, b=b, proj_ranges=(2, 3), l=l, start_canon=1)
    # step1 = LinearStep(x, [x, q], D=spa.eye(n), A=C, b=b)
    steps = [step1]

    # initsets = [BoxSet(x, zeros_n, zeros_n)]
    # initsets = [BoxSet(x, 0 * ones_n, 0 * ones_n, canon_iter=[0])]
    initsets = [L2BallSet(x, 0 * ones_n, 0, canon_iter=[0])]
    # initsets = [ConstSet(x, zeros_n)]
    # initsets = [ConstSet(x, np.ones((n, 1)))]
    # paramsets = [ConstSet(q, np.ones((m, 1)))]
    paramsets = [BoxSet(q, 10 * ones_m, 10.5 * ones_m)]

    obj = [ConvergenceResidual(x)]

    CP = CertificationProblem(K, initsets, paramsets, obj, steps)
    res = CP.solve(solver_type=solver_type, add_bounds=True,
                   add_RLT=False, TimeLimit=3600, minimize=False)
    # print('global', resg)
    # res = CP.solve(solver_type='SDP_CUSTOM')
    return res


def main():
    np.random.seed(0)
    m = 10
    n = 5
    K = 3
    A = np.random.randn(m, n)
    s = 'GLOBAL'
    # s = 'SDP'
    res1 = NNLS_cert_prob(n, m, spa.csc_matrix(A), K=K, t=.05, solver_type=s)
    # res2 = NNLS_cert_prob_two_step(n, m, spa.csc_matrix(A), K=K, t=.05, solver_type=s)
    print('combined nonneglinstep:', res1)
    # print('split nonneglinstep:', res2)


if __name__ == '__main__':
    main()
