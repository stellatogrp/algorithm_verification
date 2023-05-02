#  import certification_problem.init_set as cpi
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa

# from algocert.basic_algorithm_steps.block_step import BlockStep
# from algocert.basic_algorithm_steps.linear_step import LinearStep
from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import \
    NonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.high_level_alg_steps.nonneg_lin_step import NonNegLinStep
from algocert.init_set.box_set import BoxSet
from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.solvers.sdp_cgal_solver.lanczos import approx_min_eigvec
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


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
    # x_l = 0.5 * np.ones((n, 1))
    # x_u = np.ones((n, 1))
    # b_l = np.zeros((m, 1))
    # b_u = np.ones((m, 1))

    C = spa.bmat([[In - t * ATA, t * A.T]])
    D = spa.eye(n, n)
    b_const = spa.csc_matrix(np.zeros((n, 1)))

    # u = Iterate(n + m, name='u')
    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    # step1 = BlockStep(u, [x, b])
    # step2 = LinearStep(y, u, A=C, D=D, b=b_const)
    step1 = HighLevelLinearStep(y, [x, b], D=D, A=C, b=b_const, Dinv=D)
    step2 = NonNegProjStep(x, y)

    steps = [step1, step2]

    xset = CenteredL2BallSet(x, r=r)
    # xset = BoxSet(x, x_l, x_u)

    bset = CenteredL2BallSet(b, r=r)
    # bset = BoxSet(b, b_l, b_u)

    obj = ConvergenceResidual(x)
    CP = CertificationProblem(N, [xset], [bset], obj, steps, num_samples=1)
    CP2 = CertificationProblem(N, [xset], [bset], obj, steps, num_samples=1)
    res = CP2.solve(solver_type='SDP')
    print(res)
    # exit(0)

    # CP.problem_data = qp_problem_data
    # CP.print_cp()
    # res = CP.solve(solver_type='SDP_CGAL')
    # res = CP.solve(solver_type='SDP')

    canon = CP.canonicalize(solver_type='SDP_CGAL')
    # print(res)
    # canon.handler.solve_sketchy()
    D, xi_diffs, v_norm_diffs = canon.handler.solve(plot=True)
    print(D)
    exit(0)

    def minimum_eigvec(X):
        xi, v = spa.linalg.eigs(X, which='SR', k=1)
        return np.real(xi[0]), np.real(v)

    qt = 20
    spa_xi, spa_v = minimum_eigvec(D)
    lanc_xi, lanc_v = approx_min_eigvec(D, qt)
    print(spa_xi, lanc_xi)
    print(np.abs(spa_xi-lanc_xi))
    T = len(xi_diffs)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(range(T), xi_diffs, label='val diff', color='red')
    ax.plot(range(T), v_norm_diffs, label='vec norm diff', color='blue')

    plt.title('scgal scipy vs lanc')
    plt.xlabel('$T$')
    plt.ylabel('abs difference')
    plt.yscale('log')
    plt.ylim(1e-4, None)
    ax.axhline(y=2)
    #
    plt.legend()
    # plt.show()
    # plt.savefig('images/NNLS_multstep.pdf')


def NNLS_test_cgal(n, m, A, N=1, t=.05, xset=None, bset=None):
    N = 2
    ATA = A.T @ A
    In = spa.eye(n)
    # r = 1
    # x_l = np.zeros((n, 1))
    # x_l = 0.5 * np.ones((n, 1))
    # x_u = np.ones((n, 1))
    b_l = np.zeros((m, 1))
    b_u = np.ones((m, 1))

    C = spa.bmat([[In - t * ATA, t * A.T]])
    D = spa.eye(n, n)
    b_const = spa.csc_matrix(np.zeros((n, 1)))

    # u = Iterate(n + m, name='u')
    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    # step1 = BlockStep(u, [x, b])
    # step2 = LinearStep(y, u, A=C, D=D, b=b_const)
    step1 = HighLevelLinearStep(y, [x, b], D=D, A=C, b=b_const, Dinv=D)
    step2 = NonNegProjStep(x, y)

    steps = [step1, step2]

    xset = CenteredL2BallSet(x, r=0)
    # xset = BoxSet(x, x_l, x_u)
    # print(np.linalg.norm(xset.sample_point()))

    # bset = CenteredL2BallSet(b, r=r)
    # bset = BoxSet(b, b_l, 10 * b_u)
    bset = BoxSet(b, b_l, b_u)
    # print(bset.sample_point())
    # exit(0)

    obj = ConvergenceResidual(x)
    CP = CertificationProblem(N, [xset], [bset], obj, steps, num_samples=1)
    CP2 = CertificationProblem(N, [xset], [bset], obj, steps, num_samples=1)
    res = CP2.solve(solver_type='SDP')
    # params = CP2.solver.handler.sdp_param_outerproduct_vars
    # print(params[b].value)
    # print(np.trace(params[b].value))
    print('cp res:', res)

    # build_X

    CP.canonicalize(solver_type='SDP_CGAL', scale=True)
    CP.solve(plot=True, warmstart=False, scale_alpha=True)
    # CP.solver.handler.compare_warmstart()

    # cgal_X = CP.solve(solver_type='SDP_CGAL', plot=True, get_X=True, warmstart=False, return_resids=True)
    # print(cgal_X.shape)
    # print(np.trace(cgal_X[9:14, 9:14]))
    # print(cgal_X[9:14, -1])
    # test_b = cgal_X[9:14, -1]
    # test_x0 = cgal_X[0:3, -1]
    # test_y = cgal_X[3:6, -1]
    # print(test_y)
    # print(C @ np.hstack([test_x0, test_b]))
    # Ai = CP.solver.handler.A_matrices[1]
    # print(Ai)
    # print(np.trace(Ai @ cgal_X))


def GD_test(n, m, A, N=1, t=.05):
    ATA = A.T @ A
    In = spa.eye(n)
    r = 1
    # x_l = np.zeros((n, 1))
    # x_l = 0.5 * np.ones((n, 1))
    # x_u = np.ones((n, 1))
    # b_l = np.zeros((m, 1))
    # b_u = np.ones((m, 1))

    C = spa.bmat([[In - t * ATA, t * A.T]])
    D = spa.eye(n, n)
    b_const = spa.csc_matrix(np.zeros((n, 1)))

    # u = Iterate(n + m, name='u')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    step1 = HighLevelLinearStep(x, [x, b], D=D, A=C, b=b_const, Dinv=D)

    steps = [step1]

    xset = CenteredL2BallSet(x, r=r)
    # xset = BoxSet(x, np.zeros((n, 1)), np.zeros((n, 1)))
    # xset = BoxSet(x, x_l, x_u)

    bset = CenteredL2BallSet(b, r=r)
    # bset = BoxSet(b, b_l, b_u)
    obj = ConvergenceResidual(x)

    # CP = CertificationProblem(N, [xset], [bset], obj, steps, num_samples=1)
    CP2 = CertificationProblem(N, [xset], [bset], obj, steps, num_samples=1)
    res = CP2.solve(solver_type='SDP')
    print(res)
    # cgal_X = CP.solve(solver_type='SDP_CGAL', plot=True, get_X=True)
    # A_matrices = CP.solver.handler.A_matrices
    # b_lvals = CP.solver.handler.b_lowerbounds
    # b_uvals = CP.solver.handler.b_upperbounds
    # # for i in range(3):
    # #     print('A:', A_matrices[i])
    # #     print('bl:', b_lvals[i])
    # #     print('bu:', b_uvals[i])

    # # print(np.trace(cgal_X[0:3, 0:3]))
    # # print('testx0', cgal_X[0:3, -1])
    # # test_x0 = cgal_X[0:3, -1]
    # # test_x1 = cgal_X[3:6, -1]
    # # test_b = cgal_X[6:11, -1]
    # # print('testb', test_b)
    # # print('C @ testx0', C @ np.hstack([test_x0, test_b]))
    # # print('testx1', test_x1)

    # print('cp sdp:', res)

    # test_x1x1T = cgal_X[3:6, 3:6]
    # test_x1x0T = cgal_X[3:6, 0:3]
    # test_x1bT = cgal_X[3:6, 6:11]
    # test_x0x0T = cgal_X[0:3, 0:3]
    # test_x0bT = cgal_X[0:3, 6:11]
    # test_bbT = cgal_X[6:11, 6:11]
    # yuT = np.bmat([
    #     [test_x1x0T, test_x1bT]
    # ])
    # uuT = np.bmat([
    #     [test_x0x0T, test_x0bT],
    #     [test_x0bT.T, test_bbT]
    # ])
    # # print(yuT - C @ uuT)
    # # for i in range(len(A_matrices)):
    # #     # print('A:', A_matrices[i])
    # #     print('bl:', b_lvals[i])
    # #     print('bu:', b_uvals[i])
    # #     print(np.trace(A_matrices[i] @ cgal_X))


def NNLS_test_cgal_combined(n, m, A, N=1, t=.05):
    ATA = A.T @ A
    In = spa.eye(n)
    # r = 1
    # x_l = np.zeros((n, 1))
    # x_l = 0.5 * np.ones((n, 1))
    # x_u = np.ones((n, 1))
    b_l = np.zeros((m, 1))
    b_u = np.ones((m, 1))

    C = spa.bmat([[In - t * ATA, t * A.T]])
    # D = spa.eye(n, n)
    b_const = spa.csc_matrix(np.zeros((n, 1)))

    # u = Iterate(n + m, name='u')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    # step1 = NonNegLinStep(x, [x, b], D=D, A=C, b=b_const, Dinv=D)
    step1 = NonNegLinStep(x, [x, b], C=C, b=b_const)

    steps = [step1]

    xset = CenteredL2BallSet(x, r=0)
    # xset = BoxSet(x, np.zeros((n, 1)), np.zeros((n, 1)))
    # xset = BoxSet(x, x_l, x_u)

    # bset = CenteredL2BallSet(b, r=r)
    bset = BoxSet(b, b_l, b_u)
    obj = ConvergenceResidual(x)

    # CP = CertificationProblem(N, [xset], [bset], obj, steps, num_samples=1)
    CP2 = CertificationProblem(N, [xset], [bset], obj, steps, num_samples=1)
    res = CP2.solve(solver_type='SDP')
    # cgal_X = CP.solve(solver_type='SDP_CGAL', plot=True, get_X=True)

    print('cp res:', res)


def main():
    np.random.seed(0)
    m = 5
    n = 3
    N = 1
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    # NNLS_cert_prob(n, m, A, N=N, t=.05)
    # NNLS_test_cgal(n, m, A, N=N, t=.05)
    # GD_test(n, m, A, N=N, t=.05)
    NNLS_test_cgal_combined(n, m, A, N=N, t=.05)


if __name__ == '__main__':
    main()
