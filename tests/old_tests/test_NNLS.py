import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa

from algocert import CertificationProblem
from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import \
    NonNegProjStep
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.init_set.box_set import BoxSet
from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


def test_NNLS_SDP(N=1):
    print('----SDP----')
    m = 5
    n = 3
    r = 1

    In = spa.eye(n)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A
    # print(A)

    t = .05

    C = spa.bmat([[In - t * ATA, t * A.T]])
    D = spa.eye(n, n)
    b_const = spa.csc_matrix(np.zeros((n, 1)))

    b_l = 1
    b_u = 3

    #  u = Iterate(n + m, name='u')
    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    # step1 = BlockStep(u, [x, b])
    # step2 = LinearStep(y, u, A=C, D=D, b=b_const)
    # step3 = NonNegProjStep(x, y)
    # zeros = np.zeros((n, 1))
    # ones = np.ones((n, 1))
    # # step3 = MaxWithVecStep(x, y, l=ones)
    # steps = [step1, step2, step3]
    # # print(step1.get_output_var().name, step2.get_output_var().name, step3.get_output_var().name)

    step1 = HighLevelLinearStep(y, [x, b], D=D, A=C, b=b_const, Dinv=D)
    step2 = NonNegProjStep(x, y)
    steps = [step1, step2]

    #  Q = np.eye(n)
    #  c = np.zeros((n, 1))
    xset = CenteredL2BallSet(x, r=r)
    # xset = EllipsoidalSet(x, Q=Q, c=c)

    bset = CenteredL2BallSet(b, r=r)
    # bset = BoxSet(b, b_l, b_u)
    # c = 2 * np.ones((m, 1))
    # r = 1
    # bset = LInfBallSet(b, c, r)

    # xset = CenteredL2BallSet(x, r=r)
    x_l = -1 * np.ones((n, 1))
    x_u = np.ones((n, 1))
    xset = BoxSet(x, x_l, x_u)

    # bset = CenteredL2BallSet(b, r=r)
    b_l = np.ones((m, 1))
    b_u = 3 * np.ones((m, 1))
    bset = BoxSet(b, b_l, b_u)

    obj = ConvergenceResidual(x)
    # obj = OuterProdTrace(x)
    qp_problem_data = {'A': .5 * ATA}
    CP = CertificationProblem(N, [xset], [bset], obj, steps, qp_problem_data=qp_problem_data)

    # CP.problem_data = qp_problem_data
    # CP.print_cp()
    res = CP.solve(solver_type='SDP', add_RLT=True)
    return res


def test_NNLS_GLOBAL(N=1):
    print('--GLOBAL--')
    m = 5
    n = 3
    #  r = 1

    In = spa.eye(n)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A
    # print(A)

    t = .05

    C = spa.bmat([[In - t * ATA, t * A.T]])
    D = spa.eye(n)
    # b_const = spa.csc_matrix(np.zeros((n, 1)))
    b_const = np.zeros(n)
    l = np.zeros(n)

    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    step1 = HighLevelLinearStep(y, [x, b], D=D, A=C, b=b_const, Dinv=D)
    # step2 = NonNegProjStep(x, y)
    step2 = MaxWithVecStep(x, y, l)
    # step2 = HighLevelLinearStep(x, [y], D=D, A=D, b=b_const)

    steps = [step1]
    steps = [step1, step2]

    # xset = CenteredL2BallSet(x, r=r)
    x_l = -1 * np.ones(n)
    x_u = np.ones(n)
    xset = BoxSet(x, x_l, x_u)

    # bset = CenteredL2BallSet(b, r=r)
    b_l = np.ones(m)
    b_u = 3 * np.ones(m)
    bset = BoxSet(b, b_l, b_u)

    obj = ConvergenceResidual(x)
    # obj = OuterProdTrace(x)
    # obj = LInfConvResid(x)

    CP = CertificationProblem(N, [xset], [bset], obj, steps)

    # CP.print_cp()
    res = CP.solve(solver_type='GLOBAL')
    return res


def plot_N_vals():
    N_vals = [1, 2, 3, 4, 5, 6, 7]
    sdp_vals = []
    global_vals = []
    for N in N_vals:
        res_sdp = test_NNLS_SDP(N=N)
        res_global = test_NNLS_GLOBAL(N=N)[0]
        sdp_vals.append(res_sdp)
        global_vals.append(res_global)
    print(sdp_vals, global_vals)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(N_vals, global_vals, label='QCQP', color='red')
    ax.plot(N_vals, sdp_vals, label='SDP', color='blue')

    plt.title('Convergence residuals')
    plt.xlabel('$N$')
    plt.ylabel('maximum $||x^N - x^{N-1}||_2^2$')
    plt.yscale('log')

    plt.legend()
    plt.show()


def main():
    N = 2
    res_sdp = test_NNLS_SDP(N=N)
    res_global = test_NNLS_GLOBAL(N=N)
    print('sdp:', res_sdp, 'global:', res_global)
    # plot_N_vals(max_N=N)


if __name__ == '__main__':
    main()
