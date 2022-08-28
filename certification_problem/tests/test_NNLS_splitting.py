import numpy as np
import cvxpy as cp
import scipy.sparse as spa

from certification_problem.certification_problem import CertificationProblem
from certification_problem.variables.iterate import Iterate
from certification_problem.variables.parameter import Parameter
from certification_problem.basic_algorithm_steps.block_step import BlockStep
from certification_problem.basic_algorithm_steps.linear_step import LinearStep
from certification_problem.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from certification_problem.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep

from certification_problem.high_level_alg_steps.hl_linear_step import HighLevelLinearStep

from certification_problem.init_set.box_set import BoxSet
from certification_problem.init_set.centered_l2_ball_set import CenteredL2BallSet
from certification_problem.init_set.ellipsoidal_set import EllipsoidalSet
from certification_problem.init_set.linf_ball_set import LInfBallSet
from certification_problem.objectives.convergence_residual import ConvergenceResidual
from certification_problem.objectives.outer_prod_trace import OuterProdTrace


def test_NNLS_one_step_splitting():
    class VarLoc:
        def __init__(self, dim, start_loc):
            self.dim = dim
            self.plus_var = cp.Variable((dim, 1))
            self.plus_start_loc = start_loc
            self.plus_end_loc = start_loc + dim
            self.minus_var = cp.Variable((dim, 1))
            self.minus_start_loc = start_loc + dim
            self.minus_end_loc = start_loc + 2 * dim
            # self.slack_var = cp.Variable((dim, 1))
            # self.slack_l = start_loc + 2 * dim
            # self.slack_u = start_loc + 3 * dim

    class SlackLoc:
        def __init__(self, dim, start_loc):
            self.dim = dim
            self.var = cp.Variable((dim, 1))
            self.start_loc = start_loc
            self.end_loc = start_loc + dim

    def get_full_var(lambd, x):
        x_plus = lambd[x.plus_start_loc: x.plus_end_loc]
        x_minus = lambd[x.minus_start_loc: x.minus_end_loc]
        return x_plus - x_minus

    def get_outer_product(Lambd_mat, x, y):
        x_plus_start = x.plus_start_loc
        x_plus_end = x.plus_end_loc
        x_minus_start = x.minus_start_loc
        x_minus_end = x.minus_end_loc
        y_plus_start = y.plus_start_loc
        y_plus_end = y.plus_end_loc
        y_minus_start = y.minus_start_loc
        y_minus_end = y.minus_end_loc
        # print(x_plus_start, x_plus_end, x_minus_start, x_minus_end)
        xplus_yplusT = Lambd_mat[x_plus_start: x_plus_end, y_plus_start: y_plus_end]
        xplus_yminusT = Lambd_mat[x_plus_start: x_plus_end, y_minus_start: y_minus_end]
        xminus_yplusT = Lambd_mat[x_minus_start: x_minus_end, y_plus_start: y_plus_end]
        xminus_yminusT = Lambd_mat[x_minus_start: x_minus_end, y_minus_start: y_minus_end]
        return xplus_yplusT - xplus_yminusT - xminus_yplusT + xminus_yminusT

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
    D = spa.eye(n)
    b_const = np.zeros(n)

    x_l = -1 * np.ones(n)
    x_u = np.ones(n)
    # xset = BoxSet(x, x_l, x_u)

    b_l = np.ones(m)
    b_u = 3 * np.ones(m)
    # bset = BoxSet(b, b_l, b_u)

    # x1_plus = cp.Variable((n, 1))
    # x1_minus = cp.Variable((n, 1))
    # s_x1 = cp.Variable((n, 1))
    # y1_plus = cp.Variable((n, 1))
    # y1_minus = cp.Variable((n, 1))
    # s_y1 = cp.Variable((n, 1))
    # x0_plus = cp.Variable((n, 1))
    # x0_minus = cp.Variable((n, 1))
    # s_x0 = cp.Variable((n, 1))
    # b_plus = cp.Variable((m, 1))
    # b_minus = cp.Variable((m, 1))
    # s_b = cp.Variable((m, 1))
    x1 = VarLoc(n, 0)
    y1 = VarLoc(n, 2 * n)
    x0 = VarLoc(n, 4 * n)
    b = VarLoc(m, 6 * n)
    iter_end = 6 * n + 2 * m
    print(iter_end)
    s_x0 = SlackLoc(1, iter_end)
    s_b = SlackLoc(1, iter_end + 1)

    # lambd = cp.vstack([x1_plus, x1_minus, s_x1,
    #                    y1_plus, y1_minus, s_y1,
    #                    x0_plus, x0_minus, s_x0,
    #                    b_plus, b_minus, s_b])
    lambd = cp.vstack([x1.plus_var, x1.minus_var,
                       y1.plus_var, y1.minus_var,
                       x0.plus_var, x0.minus_var,
                       b.plus_var, b.minus_var,
                       s_x0.var, s_b.var])
    print(lambd.shape, iter_end + 1)
    lambd_dim = lambd.shape[0]
    Lambd_mat = cp.Variable((lambd_dim, lambd_dim), symmetric=True)
    full_mat = cp.bmat([
            [Lambd_mat, lambd],
            [lambd.T, np.array([[1]])]
        ])
    constraints = [full_mat >= 0, full_mat >> 0]
    # print(b.plus_start_loc, b.plus_end_loc, b.minus_start_loc, b.minus_end_loc)
    # bplus_bplusT = Lambd_mat[b.plus_start_loc: b.plus_end_loc, b.plus_start_loc: b.plus_end_loc]
    # bplus_bminusT = Lambd_mat[b.plus_start_loc: b.plus_end_loc, b.minus_start_loc: b.minus_end_loc]
    # bminus_bminusT = Lambd_mat[b.minus_start_loc: b.minus_end_loc, b.minus_start_loc: b.minus_end_loc]
    # get_outer_product(Lambd_mat, b, b)
    # constraints += [
    #     cp.trace(bplus_bplusT - 2 * bplus_bminusT + bminus_bminusT) + s_b.var == r ** 2
    # ]
    bbT_var = get_outer_product(Lambd_mat, b, b)
    x0x0T_var = get_outer_product(Lambd_mat, x0, x0)
    constraints += [
        cp.trace(bbT_var) + s_b.var == r ** 2,
        cp.trace(x0x0T_var) + s_x0.var == r ** 2
    ]
    # linstep for y1
    x0_var = get_full_var(lambd, x0)
    b_var = get_full_var(lambd, b)
    y1_var = get_full_var(lambd, y1)
    y1y1T_var = get_outer_product(Lambd_mat, y1, y1)
    x0bT_var = get_outer_product(Lambd_mat, x0, b)
    constraints += [
        y1_var == C @ cp.vstack([x0_var, b_var]),
        y1y1T_var == C @ cp.bmat([
            [x0x0T_var, x0bT_var],
            [x0bT_var.T, bbT_var]
        ])@ C.T,
    ]

    #proj for x1
    x1_var = get_full_var(lambd, x1)
    x1x1T_var = get_outer_product(Lambd_mat, x1, x1)
    x1y1T_var = get_outer_product(Lambd_mat, x1, y1)
    constraints += [
        x1_var >= 0, x1x1T_var >= 0,
        cp.diag(x1x1T_var - x1y1T_var) == 0,
        # Lambd_mat[x1.plus_start_loc: x1.plus_end_loc, x1.minus_start_loc: x1.minus_end_loc] == 0
    ]

    x1x0T_var = get_outer_product(Lambd_mat, x1, x0)
    # constraints += [x1x0T_var >= 0]
    # obj = cp.Maximize(cp.trace(y1y1T_var))
    obj = cp.Maximize(cp.trace(x1x1T_var - 2 * x1x0T_var + x0x0T_var))
    prob = cp.Problem(obj, constraints)
    res = prob.solve()
    print(res)
    # print(Lambd_mat.value, y1_var.value)


def test_NNLS_GLOBAL(N=1):
    print('--GLOBAL--')
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
    D = spa.eye(n)
    # b_const = spa.csc_matrix(np.zeros((n, 1)))
    b_const = np.zeros(n)

    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    step1 = HighLevelLinearStep(y, [x, b], D=D, A=C, b=b_const)
    step2 = NonNegProjStep(x, y)
    # step2 = HighLevelLinearStep(x, [y], D=D, A=D, b=b_const)

    steps = [step1]
    steps = [step1, step2]

    # xset = CenteredL2BallSet(x, r=r)
    x_l = -1 * np.ones(n)
    x_u = np.ones(n)
    # xset = BoxSet(x, x_l, x_u)
    xset = CenteredL2BallSet(x, r=1)

    # bset = CenteredL2BallSet(b, r=r)
    b_l = np.ones(m)
    b_u = 3 * np.ones(m)
    bset = BoxSet(b, b_l, b_u)
    bset = CenteredL2BallSet(b, r=1)

    obj = ConvergenceResidual(x)
    # obj = OuterProdTrace(x)

    CP = CertificationProblem(N, [xset], [bset], obj, steps)

    # CP.print_cp()
    res = CP.solve(solver_type='GLOBAL')
    return res


def main():
    test_NNLS_one_step_splitting()
    N = 1
    test_NNLS_GLOBAL(N=N)


if __name__ == '__main__':
    main()
