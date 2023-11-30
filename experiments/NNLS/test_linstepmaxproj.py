import numpy as np
import scipy.sparse as spa

from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.linear_max_proj_step import LinearMaxProjStep
from algocert.high_level_alg_steps.linear_step import LinearStep
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
    # TODO split into two steps and compare to see
    # step1 = LinearMaxProjStep(x, [x, q], A=C, b=b, proj_ranges=(0, 5), l=l, start_canon=1)
    step1 = LinearMaxProjStep(x, [x, q], A=C, b=b, l=l, start_canon=1)
    # step1 = LinearStep(x, [x, q], D=spa.eye(n), A=C, b=b, Dinv=spa.eye(n), start_canon=1)
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
    # res = CP.solve(solver_type=solver_type, add_bounds=True,
                #    add_RLT=False, TimeLimit=3600, minimize=False)
    # if solver_type == 'GLOBAL':
    #     xmat = CP.solver.handler.get_iterate_var_map()[x].X
    #     qval = CP.solver.handler.get_param_var_map()[q].X
    #     print('x:', xmat, 'q:', qval)
    #     print(np.linalg.norm(xmat[K] - xmat[K-1]) ** 2)
    #     x_curr = xmat[0]
        # print('simulating')
        # for i in range(1, K+1):
        #     rhs = np.hstack([x_curr, qval])
        #     x_out = np.maximum(C.todense() @ rhs, l.reshape(-1, ))
        #     x_out = np.squeeze(np.asarray(x_out))
        #     print(x_out, np.linalg.norm(x_out - x_curr) ** 2)
        #     x_curr = x_out
        # print('x upper:', CP.solver.handler.iterate_to_upper_bound_map[x])
    # print('global', resg)
    res = CP.solve(solver_type='SDP_CUSTOM')
    return res


def NNLS_twostep(n, m, A, K=1, t=.05, solver_type='SDP'):
    ATA = A.T @ A
    In = spa.eye(n)
    zeros_n = np.zeros((n, 1))
    # zeros_m = np.zeros((m, 1))
    ones_m = np.ones((m, 1))
    ones_n = np.ones((n, 1))

    C = spa.bmat([[In - t * ATA, t * A.T]])
    b = zeros_n

    x = Iterate(n, name='x')
    y = Iterate(n, name='y')
    q = Parameter(m, name='q')

    l = 0.1 * np.ones((n, 1))

    step1 = LinearStep(y, [x, q], D=spa.eye(n), A=C, b=b, Dinv=spa.eye(n), start_canon=1)
    step2 = MaxWithVecStep(x, y, l=l)

    steps = [step1, step2]
    initsets = [L2BallSet(x, 0 * ones_n, 0, canon_iter=[0])]
    paramsets = [BoxSet(q, 10 * ones_m, 10.5 * ones_m)]

    obj = [ConvergenceResidual(x)]

    CP = CertificationProblem(K, initsets, paramsets, obj, steps)
    # res = CP.solve(solver_type=solver_type, add_bounds=True,
                #    add_RLT=False, TimeLimit=3600, minimize=False)
    # # print('global', resg)
    # if solver_type == 'GLOBAL':
    #     xmat = CP.solver.handler.get_iterate_var_map()[x].X
    #     qval = CP.solver.handler.get_param_var_map()[q].X
    #     print('x:', xmat, 'q:', qval)
    #     print('y:', CP.solver.handler.get_iterate_var_map()[y].X)
    #     print(np.linalg.norm(xmat[K] - xmat[K-1]) ** 2)
    #     x_curr = xmat[0]
    #     for i in range(1, K+1):
    #         rhs = np.hstack([x_curr, qval])
    #         x_out = np.maximum(C.todense() @ rhs, l.reshape(-1, ))
    #         x_out = np.squeeze(np.asarray(x_out))
    #         print(x_out, np.linalg.norm(x_out - x_curr) ** 2)
    #         x_curr = x_out
    res = CP.solve(solver_type='SDP_CUSTOM')
    return res


def main():
    np.random.seed(0)
    m = 10
    n = 5
    K = 2
    A = np.random.randn(m, n)
    s = 'GLOBAL'
    # s = 'SDP'

    eigs = np.linalg.eigvals(A.T @ A)
    np.min(eigs)
    np.max(eigs)

    # t = 2 / (mu + L)
    t = .05
    print(t)

    # res1 = NNLS_cert_prob(n, m, spa.csc_matrix(A), K=K, t=t, solver_type=s)
    # print('combined nonneglinstep:', res1)

    res2 = NNLS_twostep(n, m, spa.csc_matrix(A), K=K, t=t, solver_type=s)
    print('two steps:', res2)



if __name__ == '__main__':
    main()
