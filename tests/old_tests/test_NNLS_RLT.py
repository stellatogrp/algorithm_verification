import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from algocert.basic_algorithm_steps.block_step import BlockStep
from algocert.basic_algorithm_steps.linear_step import LinearStep
from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.init_set.box_set import BoxSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


def lin_bound_map(l, u, A):
    A = A.toarray()
    (m, n) = A.shape
    l_out = np.zeros(m)
    u_out = np.zeros(m)
    for i in range(m):
        lower = 0
        upper = 0
        for j in range(n):
            if A[i][j] >= 0:
                lower += A[i][j] * l[j]
                upper += A[i][j] * u[j]
            else:
                lower += A[i][j] * u[j]
                upper += A[i][j] * l[j]
        l_out[i] = lower
        u_out[i] = upper

    return np.reshape(l_out, (m, 1)), np.reshape(u_out, (m, 1))


def nonneg_proj_bound_map(l, u, n):
    l_out = np.reshape(np.zeros(n), (n, 1))
    l_out = np.maximum(l, l_out)
    u_out = np.maximum(u, 0)

    # print(l_out, u_out)
    return np.reshape(l_out, (n, 1)), np.reshape(u_out, (n, 1))


def RLT_constraints(xyT, x, lx, ux, y, ly, uy):
    return [
        xyT - x @ ly.T - lx @ y.T + lx @ ly.T >= 0,
        x @ uy.T - xyT - lx @ uy.T + lx @ y.T >= 0,
        ux @ y.T - ux @ ly.T - xyT + x @ ly.T >= 0,
        ux @ uy.T - ux @ y.T - x @ uy.T + xyT >= 0,
    ]


def test_NNLS_SDPRLT(m, n, N, t, r, A):
    class VarBounds:
        def __init__(self, var, var_outer, l, u):
            self.var = var
            self.var_outer = var_outer
            self.l = l
            self.u = u
    print('--------RLT--------')
    ATA = A.T @ A
    In = spa.eye(n)

    C = spa.bmat([[In - t * ATA, t * A.T]])

    x0_l = -1 * np.ones((n, 1))
    x0_u = np.ones((n, 1))

    b_l = 1 * np.ones((m, 1))
    b_u = 3 * np.ones((m, 1))

    b = cp.Variable((m, 1))
    x0 = cp.Variable((n, 1))
    y1 = cp.Variable((n, 1))
    x1 = cp.Variable((n, 1))
    y2 = cp.Variable((n, 1))
    x2 = cp.Variable((n, 1))

    bbT = cp.Variable((m, m))

    x0x0T = cp.Variable((n, n))
    x0bT = cp.Variable((n, m))

    y1y1T = cp.Variable((n, n))
    y1x0T = cp.Variable((n, n))
    y1bT = cp.Variable((n, m))

    x1x1T = cp.Variable((n, n))
    x1y1T = cp.Variable((n, n))
    x1x0T = cp.Variable((n, n))
    x1bT = cp.Variable((n, m))

    y2y2T = cp.Variable((n, n))
    y2x1T = cp.Variable((n, n))
    y2bT = cp.Variable((n, m))

    x2x2T = cp.Variable((n, n))
    x2y2T = cp.Variable((n, n))
    x2x1T = cp.Variable((n, n))
    x2bT = cp.Variable((n, m))

    constraints = [
        cp.reshape(cp.diag(x0x0T), (n, 1)) <= cp.multiply(x0_l + x0_u, x0) - cp.multiply(x0_l, x0_u),
        # cp.trace(x0x0T) <= 1,
        cp.reshape(cp.diag(bbT), (m, 1)) <= cp.multiply(b_l + b_u, b) - cp.multiply(b_l, b_u)
    ]

    # Linstep 1
    u1 = cp.vstack([x0, b])
    u1u1T = cp.bmat([
        [x0x0T, x0bT],
        [x0bT.T, bbT]
    ])
    y1u1T = cp.hstack([y1x0T, y1bT])
    # print((y1u1T).shape, (C @ u1u1T).shape)

    constraints += [
        y1 == C @ u1, y1y1T == C @ u1u1T @ C.T, y1u1T == C @ u1u1T,
        cp.bmat([
            [y1y1T, y1u1T, y1],
            [y1u1T.T, u1u1T, u1],
            [y1.T, u1.T, np.array([[1]])]
        ]) >> 0,
    ]

    # Nonneg proj 1
    x1u1T = cp.hstack([x1x0T, x1bT])
    constraints += [
        x1 >= 0, x1x1T >= 0, x1 >= y1, cp.diag(x1x1T - x1y1T) == 0,
        x1y1T == x1u1T @ C.T,
        cp.bmat([
            [x1x1T, x1y1T, x1],
            [x1y1T.T, y1y1T, y1],
            [x1.T, y1.T, np.array([[1]])]
        ]) >> 0,
    ]

    # Linstep 2
    u2 = cp.vstack([x1, b])
    u2u2T = cp.bmat([
        [x1x1T, x1bT],
        [x1bT.T, bbT]
    ])
    y2u2T = cp.hstack([y2x1T, y2bT])
    # print((y1u1T).shape, (C @ u1u1T).shape)

    constraints += [
        y2 == C @ u2, y2y2T == C @ u2u2T @ C.T, y2u2T == C @ u2u2T,
        cp.bmat([
            [y2y2T, y2u2T, y2],
            [y2u2T.T, u2u2T, u2],
            [y2.T, u2.T, np.array([[1]])]
        ]) >> 0,
    ]

    # Nonneg proj 2
    x2u2T = cp.hstack([x2x1T, x2bT])
    constraints += [
        x2 >= 0, x2x2T >= 0, x2 >= y2, cp.diag(x2x2T - x2y2T) == 0,
        x2y2T == x2u2T @ C.T,
        cp.bmat([
            [x2x2T, x2y2T, x2],
            [x2y2T.T, y2y2T, y2],
            [x2.T, y2.T, np.array([[1]])]
        ]) >> 0,
    ]

    # final obj
    constraints += [
        cp.bmat([
            [x1x1T, x1x0T, x1],
            [x1x0T.T, x0x0T, x0],
            [x1.T, x0.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [x2x2T, x2x1T, x2],
            [x2x1T.T, x1x1T, x1],
            [x2.T, x1.T, np.array([[1]])]
        ]) >> 0,
    ]

    x2x0T = cp.Variable((n, n))
    constraints += [
        cp.bmat([
            [x2x2T, x2x0T, x2],
            [x2x0T.T, x0x0T, x0],
            [x2.T, x0.T, np.array([[1]])]
        ]) >> 0,
    ]

    # RLT
    RLT_add = True
    if RLT_add:
        #  b_vb = VarBounds(b, bbT, b_l, b_u)
        #  x0_vb = VarBounds(x0, x0x0T, x0_l, x0_u)
        u1_l = np.vstack([x0_l, b_l])
        u1_u = np.vstack([x0_u, b_u])
        #  u1_vb = VarBounds(u1, u1u1T, u1_l, u1_u)
        y1_l, y1_u = lin_bound_map(u1_l, u1_u, C)
        #  y1_vb = VarBounds(y1, y1y1T, y1_l, y1_u)
        x1_l, x1_u = nonneg_proj_bound_map(y1_l, y1_u, n)

        u2_l = np.vstack([x1_l, b_l])
        u2_u = np.vstack([x1_u, b_u])
        y2_l, y2_u = lin_bound_map(u2_l, u2_u, C)
        x2_l, x2_u = nonneg_proj_bound_map(y2_l, y2_u, n)

        # RLT constraints with iterates and b
        constraints += RLT_constraints(x0bT, x0, x0_l, x0_u, b, b_l, b_u)
        # constraints += RLT_constraints(u1bT)
        constraints += RLT_constraints(y1bT, y1, y1_l, y1_u, b, b_l, b_u)
        constraints += RLT_constraints(x1bT, x1, x1_l, x1_u, b, b_l, b_u)

        # RLT constraints, vec with themselves (is this necessary ?)
        constraints += RLT_constraints(x0x0T, x0, x0_l, x0_u, x0, x0_l, x0_u)
        constraints += RLT_constraints(u1u1T, u1, u1_l, u1_u, u1, u1_l, u1_u)
        constraints += RLT_constraints(y1y1T, y1, y1_l, y1_u, y1, y1_l, y1_u)
        constraints += RLT_constraints(x1x1T, x1, x1_l, x1_u, x1, x1_l, x1_u)

        constraints += RLT_constraints(x1y1T, x1, x1_l, x1_u, y1, y1_l, y1_u)
        constraints += RLT_constraints(x1x0T, x1, x1_l, x1_u, x0, x0_l, x0_u)
        constraints += RLT_constraints(y1u1T, y1, y1_l, y1_u, u1, u1_l, u1_u)

        constraints += RLT_constraints(x2x1T, x2, x2_l, x2_u, x1, x1_l, x1_u)
        constraints += RLT_constraints(x2x2T, x2, x2_l, x2_u, x2, x2_l, x2_u)

    obj = cp.trace(x2x2T - 2 * x2x1T + x1x1T)
    # obj = cp.trace(x1x1T - 2 * x1x0T + x0x0T)
    prob = cp.Problem(cp.Maximize(obj), constraints)
    res = prob.solve()
    print(res)


def test_NNLS_SDP(m, n, N, t, r, A):
    print('----SDP----')
    ATA = A.T @ A
    In = spa.eye(n)

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
    # print(step1.get_output_var().name, step2.get_output_var().name, step3.get_output_var().name)

    # xset = CenteredL2BallSet(x, r=r)
    x_l = -1 * np.ones((n, 1))
    x_u = np.ones((n, 1))
    xset = BoxSet(x, x_l, x_u)

    b_l = 1 * np.ones((m, 1))
    b_u = 3 * np.ones((m, 1))
    bset = BoxSet(b, b_l, b_u)

    obj = ConvergenceResidual(x)
    qp_problem_data = {'A': .5 * ATA}
    CP = CertificationProblem(N, [xset], [bset], obj, steps, qp_problem_data=qp_problem_data)

    # CP.problem_data = qp_problem_data
    # CP.print_cp()
    CP.solve(solver_type='SDP')


def test_NNLS_GLOBAL(m, n, N, t, r, A):

    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')
    # print(y, x, b)

    ATA = A.T @ A
    In = spa.eye(n)
    C = spa.bmat([[In - t * ATA, t * A.T]])
    zeros = np.zeros(n)

    x_l = -1 * np.ones(n)
    x_u = np.ones(n)

    # xset = CenteredL2BallSet(x, r)
    xset = BoxSet(x, x_l, x_u)

    # bset = CenteredL2BallSet(b, r=r)
    b_l = 1 * np.ones(m)
    b_u = 3 * np.ones(m)
    bset = BoxSet(b, b_l, b_u)

    step1 = HighLevelLinearStep(y, [x, b], D=In, A=C, b=zeros)
    step2 = NonNegProjStep(x, y)

    steps = [step1, step2]

    obj = ConvergenceResidual(x)
    CP = CertificationProblem(N, [xset], [bset], obj, steps)
    res = CP.solve(solver_type='GLOBAL')
    res


def test_linbound_map():
    l = np.array([[-3], [-3]])
    u = np.array([[-1], [-1]])

    A = np.array([[1, 1],
                  [-1, -1],
                  [-1, 2]])
    A = spa.csc_matrix(A)
    lower, upper = lin_bound_map(l, u, A)
    print('lower:', lower)
    print('upper:', upper)


def test_nonneg_proj_map():
    n = 3
    l = np.array([[1], [-1], [-2]])
    u = np.array([[2], [-1], [3]])

    lower, upper = nonneg_proj_bound_map(l, u, n)
    print('lower:', lower)
    print('upper:', upper)


def main():
    m = 5
    n = 3
    N = 2
    t = .05
    r = 1

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)

    # test_linbound_map()
    # test_nonneg_proj_map()
    # test_NNLS_SDP(m, n, N, t, r, A)
    test_NNLS_SDPRLT(m, n, N, t, r, A)
    # test_NNLS_GLOBAL(m, n, N, t, r, A)


if __name__ == '__main__':
    main()
