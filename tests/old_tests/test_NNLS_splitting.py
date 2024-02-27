import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from algoverify import VerificationProblem
from algoverify.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algoverify.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algoverify.init_set.box_set import BoxSet
from algoverify.init_set.centered_l2_ball_set import CenteredL2BallSet
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.variables.iterate import Iterate
from algoverify.variables.parameter import Parameter


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


def get_plus_part(lambd, x):
    # print(x.plus_start_loc, x.plus_end_loc)
    x_plus = lambd[x.plus_start_loc: x.plus_end_loc]
    return x_plus


def get_plus_plus_outer_prod(Lambd_mat, x, y):
    x_plus_start = x.plus_start_loc
    x_plus_end = x.plus_end_loc
    y_plus_start = y.plus_start_loc
    y_plus_end = y.plus_end_loc
    xplus_yplusT = Lambd_mat[x_plus_start: x_plus_end, y_plus_start: y_plus_end]
    return xplus_yplusT


def get_plus_minus_outer_prod(Lambd_mat, x, y):
    x_plus_start = x.plus_start_loc
    x_plus_end = x.plus_end_loc
    y_minus_start = y.minus_start_loc
    y_minus_end = y.minus_end_loc
    xplus_yminusT = Lambd_mat[x_plus_start: x_plus_end, y_minus_start: y_minus_end]
    return xplus_yminusT


def get_minus_minus_outer_prod(Lambd_mat, x, y):
    x_minus_start = x.minus_start_loc
    x_minus_end = x.minus_end_loc
    y_minus_start = y.minus_start_loc
    y_minus_end = y.minus_end_loc
    xminus_yminusT = Lambd_mat[x_minus_start: x_minus_end, y_minus_start: y_minus_end]
    return xminus_yminusT


def get_nearest_rank_one_mat(X):
    U, Sigma, V = np.linalg.svd(X, full_matrices=False)
    return Sigma[0] * np.outer(U.T[0], V[0])


def mat_to_vec_sqrt(X):
    return np.sqrt(np.diag(X))


def test_NNLS_one_step_splitting():
    print('splitting normal')
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
    #  D = spa.eye(n)
    #  b_const = np.zeros(n)

    #  x_l = -1 * np.ones(n)
    #  x_u = np.ones(n)
    # xset = BoxSet(x, x_l, x_u)

    #  b_l = np.ones(m)
    #  b_u = 3 * np.ones(m)
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
        ]) @ C.T,
    ]

    # proj for x1
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

    # resid_var = x1_var - x0_var
    # M = 100
    # t = cp.Variable()
    # z_pos = cp.Variable((n, 1))
    # z_neg = cp.Variable((n, 1))
    # constraints += [
    #     resid_var <= t,
    #     t <= resid_var + M * (1-z_pos),
    #     0 <= z_pos,
    #     z_pos <= 1,
    #     cp.sum(z_pos) == 1,
    #
    #     -resid_var <= t,
    #     t <= -resid_var + M * (1 - z_neg),
    #     0 <= z_neg,
    #     z_neg <= 1,
    #     cp.sum(z_neg) == 1,
    # ]

    # obj = cp.Maximize(cp.trace(x1x1T_var))
    # obj = cp.Maximize(t)
    # constraints += [cp.trace(x1x1T_var - 2 * x1x0T_var + x0x0T_var) <= 1]
    obj = cp.Maximize(cp.trace(x1x1T_var - 2 * x1x0T_var + x0x0T_var))
    prob = cp.Problem(obj, constraints)
    res = prob.solve()
    print(res)
    #  temp = np.bmat([
    #      [x1x1T_var.value, x1x0T_var.value, x1_var.value],
    #      [x1x0T_var.T.value, x0x0T_var.value, x0_var.value],
    #      [x1_var.T.value, x0_var.T.value, np.array([[1]])]
    #  ])
    # print(temp)
    # print('eigenvalues:', np.round(np.linalg.eigvals(temp), 4))
    # print(np.round(np.linalg.eigvals(Lambd_mat.value), 4))
    # print('x1x1T eigvals:', np.round(np.linalg.eigvals(x1x1T_var.value), 4))
    # print('x1x1T:', np.round(x1x1T_var.value, 4))
    # x1_rank1_approx = get_nearest_rank_one_mat(x1x1T_var.value)
    # print(np.round(x1_rank1_approx, 4))
    # x1_test = mat_to_vec_sqrt(x1_rank1_approx)
    # print(np.round(x1_test, 4))
    #
    # print('x0x0T eigvals:', np.round(np.linalg.eigvals(x0x0T_var.value), 4))
    # x0_rank1_approx = get_nearest_rank_one_mat(x0x0T_var.value)
    # print('x0x0T:', x0x0T_var.value)
    # x0_test = mat_to_vec_sqrt(x0_rank1_approx)
    # x0_test[0] = -x0_test[0]
    # print(np.round(x0_test, 4))
    # print(np.linalg.norm(x1_test-x0_test) ** 2)
    # print(cp.trace(x1x1T_var).value)


def test_NNLS_one_step_split_only_y():
    print('splitting only y')
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
    #  D = spa.eye(n)
    #  b_const = np.zeros(n)

    y1 = VarLoc(n, 0)
    x0 = VarLoc(n, 2 * n)
    b = VarLoc(m, 4 * n)
    iter_end = 4 * n + 2 * m
    print(iter_end)
    s_x0 = SlackLoc(1, iter_end)
    s_b = SlackLoc(1, iter_end + 1)

    lambd = cp.vstack([y1.plus_var, y1.minus_var,
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

    bbT_var = get_outer_product(Lambd_mat, b, b)
    x0x0T_var = get_outer_product(Lambd_mat, x0, x0)
    constraints += [
        cp.trace(bbT_var) + s_b.var == r ** 2,
        cp.trace(x0x0T_var) + s_x0.var == r ** 2
    ]
    # linstep for y1
    # x0_var = get_full_var(lambd, x0)
    # b_var = get_full_var(lambd, b)
    # y1_var = get_full_var(lambd, y1)
    # y1y1T_var = get_outer_product(Lambd_mat, y1, y1)
    # x0bT_var = get_outer_product(Lambd_mat, x0, b)
    # constraints += [
    #     y1_var == C @ cp.vstack([x0_var, b_var]),
    #     y1y1T_var == C @ cp.bmat([
    #         [x0x0T_var, x0bT_var],
    #         [x0bT_var.T, bbT_var]
    #     ]) @ C.T,
    # ]
    x0_var = get_full_var(lambd, x0)
    x0x0T_var = get_outer_product(Lambd_mat, x0, x0)
    x0bT_var = get_outer_product(Lambd_mat, x0, b)
    #  x0plus_var = get_plus_part(lambd, x0)
    #  x0plus_x0plusT_var = get_plus_plus_outer_prod(Lambd_mat, x0, x0)
    b_var = get_full_var(lambd, b)
    y1_var = get_full_var(lambd, y1)
    y1y1T_var = get_outer_product(Lambd_mat, y1, y1)
    y1plus_y1plusT_var = get_plus_plus_outer_prod(Lambd_mat, y1, y1)
    #  y1minus_y1minusT_var = get_minus_minus_outer_prod(Lambd_mat, y1, y1)

    #  x0plus_bplusT_var = get_plus_plus_outer_prod(Lambd_mat, x0, b)
    #  x0plus_bminusT_var = Lambd_mat[x0.plus_start_loc: x0.plus_end_loc, b.minus_start_loc: b.minus_end_loc]
    #  x0plus_bT_var = x0plus_bplusT_var - x0plus_bminusT_var
    # constraints += [
    #     y1_var == C @ cp.vstack([x0plus_var, b_var]),
    #     y1y1T_var == C @ cp.bmat([
    #         [x0plus_x0plusT_var, x0plus_bT_var],
    #         [x0plus_bT_var.T, bbT_var]
    #     ]) @ C.T,
    #     # y1y1T_var == y1plus_y1plusT_var + y1minus_y1minusT_var,
    # ]
    constraints += [
        y1_var == C @ cp.vstack([x0_var, b_var]),
        y1y1T_var == C @ cp.bmat([
            [x0x0T_var, x0bT_var],
            [x0bT_var.T, bbT_var]
        ]) @ C.T,
    ]
    # key idea, the 'projection' of y1 has already happened because its just y1plus
    # so, just need to enforce the complementarity constraint
    y1plus_y1minus_var = get_plus_minus_outer_prod(Lambd_mat, y1, y1)
    constraints += [y1plus_y1minus_var == 0]

    # for the objective, x1 is just y1plus
    # so tr(x1x1T - x1x0T + x0x0T) becomes
    # tr(y1plus_y1plusT - y1plus_x0plusT + x0plus_x0plusT)
    y1plus_y1plusT_var = get_plus_plus_outer_prod(Lambd_mat, y1, y1)
    #  y1plus_x0plusT_var = get_plus_plus_outer_prod(Lambd_mat, y1, x0)
    #  y1plus_x0minusT_var = get_plus_minus_outer_prod(Lambd_mat, y1, x0)
    #  y1plus_x0T_var = y1plus_x0plusT_var - y1plus_x0minusT_var
    # obj = cp.Maximize(cp.trace(y1plus_y1plusT_var - 2 * y1plus_x0plusT_var + x0plus_x0plusT_var))

    # obj = cp.Maximize(cp.trace(y1plus_y1plusT_var - 2 * y1plus_x0T_var + x0x0T_var))
    obj = cp.Maximize(cp.trace(y1plus_y1plusT_var))
    prob = cp.Problem(obj, constraints)
    res = prob.solve()
    print(res)


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
    #  x_l = -1 * np.ones(n)
    #  x_u = np.ones(n)
    # xset = BoxSet(x, x_l, x_u)
    xset = CenteredL2BallSet(x, r=r)

    # bset = CenteredL2BallSet(b, r=r)
    b_l = np.ones(m)
    b_u = 3 * np.ones(m)
    bset = BoxSet(b, b_l, b_u)
    bset = CenteredL2BallSet(b, r=r)

    obj = ConvergenceResidual(x)
    # obj = OuterProdTrace(x)
    # obj = LInfConvResid(x)

    CP = VerificationProblem(N, [xset], [bset], obj, steps)

    # CP.print_cp()
    res = CP.solve(solver_type='GLOBAL')
    return res


def main():
    test_NNLS_one_step_splitting()
    # test_NNLS_one_step_split_only_y()
    #  N = 1
    # test_NNLS_GLOBAL(N=N)


if __name__ == '__main__':
    main()
