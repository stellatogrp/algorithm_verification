import numpy as np
import scipy.sparse as spa

from algoverify.basic_algorithm_steps.block_step import BlockStep
from algoverify.basic_algorithm_steps.linear_step import LinearStep
from algoverify.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algoverify.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algoverify.init_set.box_set import BoxSet
from algoverify.objectives.outer_prod_trace import OuterProdTrace
from algoverify.variables.iterate import Iterate
from algoverify.variables.parameter import Parameter
from algoverify.verification_problem import VerificationProblem


def test_NNLS_SDP(N=1):
    print('----SDP----')
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
    D = spa.eye(n, n)
    I_negI = spa.bmat([[D, -D]])
    b_const = spa.csc_matrix(np.zeros((n, 1)))

    zblock = Iterate(n, name='zblock')
    z = Iterate(n, name='z')
    u = Iterate(n + m, name='u')
    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    wblock = Iterate(2 * n, name='wblock')
    w = Iterate(n, name='w')
    b = Parameter(m, name='b')

    # step1 = BlockStep(u, [x, b])
    # step2 = LinearStep(y, u, A=C, D=D, b=b_const)
    # step3 = NonNegProjStep(x, y)
    #
    # steps = [step1, step2, step3]
    step1 = BlockStep(zblock, [x])
    step2 = LinearStep(z, zblock, A=D, D=D, b=b_const)
    step3 = BlockStep(u, [z, b])
    step4 = LinearStep(y, u, A=C, D=D, b=b_const)
    step5 = NonNegProjStep(x, y)
    step6 = BlockStep(wblock, [x, z])
    step7 = LinearStep(w, wblock, A=I_negI, D=D, b=b_const)

    steps = [step1, step2, step3, step4, step5, step6, step7]

    x_l = -1 * np.ones((n, 1))
    x_u = np.ones((n, 1))
    xset = BoxSet(x, x_l, x_u)

    # bset = CenteredL2BallSet(b, r=r)
    b_l = np.ones((m, 1))
    b_u = 3 * np.ones((m, 1))
    bset = BoxSet(b, b_l, b_u)

    # obj = ConvergenceResidual(x)
    obj = OuterProdTrace(w)
    qp_problem_data = {'A': .5 * ATA}
    CP = VerificationProblem(N, [xset], [bset], obj, steps, qp_problem_data=qp_problem_data)

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
    I_negI = spa.bmat([[D, -D]])
    # b_const = spa.csc_matrix(np.zeros((n, 1)))
    b_const = np.zeros(n)

    z = Iterate(n, name='z')
    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    w = Iterate(n, name='w')
    b = Parameter(m, name='b')

    # step1 = HighLevelLinearStep(y, [x, b], D=D, A=C, b=b_const)
    # step2 = NonNegProjStep(x, y)
    # steps = [step1, step2]
    step1 = HighLevelLinearStep(z, [x], D=D, A=D, b=b_const)
    step2 = HighLevelLinearStep(y, [z, b], D=D, A=C, b=b_const)
    step3 = NonNegProjStep(x, y)
    step4 = HighLevelLinearStep(w, [x, z], D=D, A=I_negI, b=b_const)

    steps = [step1, step2, step3, step4]

    # xset = CenteredL2BallSet(x, r=r)
    x_l = -1 * np.ones(n)
    x_u = np.ones(n)
    xset = BoxSet(x, x_l, x_u)

    # bset = CenteredL2BallSet(b, r=r)
    b_l = np.ones(m)
    b_u = 3 * np.ones(m)
    bset = BoxSet(b, b_l, b_u)

    # obj = ConvergenceResidual(x)
    obj = OuterProdTrace(w)
    # obj = OuterProdTrace(x)

    CP = VerificationProblem(N, [xset], [bset], obj, steps)

    # CP.print_cp()
    res = CP.solve(solver_type='GLOBAL')
    return res


def main():
    N = 3
    res_sdp = test_NNLS_SDP(N=N)
    res_sdp
    # res_global = test_NNLS_GLOBAL(N=N)
    # print('sdp:', res_sdp, 'global:', res_global)


if __name__ == '__main__':
    main()
