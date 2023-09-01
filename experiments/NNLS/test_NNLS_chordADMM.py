#  import certification_problem.init_set as cpi
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
from tqdm import trange

# from algocert.basic_algorithm_steps.block_step import BlockStep
# from algocert.basic_algorithm_steps.linear_step import LinearStep
from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep

# from algocert.high_level_alg_steps.nonneg_lin_step import NonNegLinStep
from algocert.init_set.box_set import BoxSet
from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
from algocert.objectives.convergence_residual import ConvergenceResidual

# from algocert.solvers.sdp_cgal_solver.lanczos import approx_min_eigvec
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


# The vec function as documented in api/cones
def vec(S):
    n = S.shape[0]
    S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n)] /= np.sqrt(2)
    return S[np.triu_indices(n)]


# The mat function as documented in api/cones
def mat(s):
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S


def box_proj(x, l, u):
    # Pi(x) = min(u, max(x, l))
    return np.minimum(u, np.maximum(x, l))


def NNLS_test_cgal(n, m, A, N=1, t=.05, xset=None, bset=None):
    ATA = A.T @ A
    In = spa.eye(n)
    r = 1
    # x_l = np.zeros((n, 1))
    # x_l = 0.5 * np.ones((n, 1))
    # x_u = np.ones((n, 1))
    b_l = np.zeros((m, 1))
    b_u = np.ones((m, 1))

    C = spa.bmat([[In - t * ATA, t * A.T]])
    D = spa.eye(n, n)
    b_const = spa.csc_matrix(np.zeros((n, 1)))

    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')

    # step1 = BlockStep(u, [x, b])
    # step2 = LinearStep(y, u, A=C, D=D, b=b_const)
    step1 = HighLevelLinearStep(y, [x, b], D=D, A=C, b=b_const, Dinv=D)
    step2 = NonNegProjStep(x, y)

    steps = [step1, step2]

    xset = CenteredL2BallSet(x, r=0)
    # xset = BoxSet(x, np.zeros((n, 1)), np.zeros((n, 1)))
    # xset = BoxSet(x, x_l, x_u)

    bset = CenteredL2BallSet(b, r=r)
    bset = BoxSet(b, b_l, b_u)
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
    handler = CP.solver.handler
    A_mats = handler.A_matrices
    l = np.array(handler.b_lowerbounds)
    u = np.array(handler.b_upperbounds)
    A_vecs = []
    for A in A_mats:
        # A_dense = A.todense()
        # print(A)
        # print(A_dense, vec(A_dense))
        # print(vec(A.todense()).shape)
        A_vecs.append(vec(A.todense()))
    A = np.vstack(A_vecs)
    # test = np.ones(37)
    # print(np.linalg.norm(test - box_proj(test, l, u)))
    # print(box_proj(test, l, u))
    C_mat = handler.C_matrix.todense()
    c = vec(C_mat)
    res = handler.test_with_cvxpy()
    print('cvxpy:', res)
    ADMM(c, A, l, u, res=res)


def vec_psd_proj(x):
    x_mat = mat(x)
    x_mat_proj = psd_proj(x_mat)
    return vec(x_mat_proj)


def psd_proj(X):
    # # print(check_symmetric(X))
    # # eigvals, eigvecs = np.linalg.eig(X)
    # # print(eigvals)
    # # eigvals_clipped = np.maximum(eigvals, 0)
    # # print(eigvals_clipped)
    # # output = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    # # print(np.linalg.eigvals(output))
    # # print(np.linalg.norm(X-output))

    # eigvals, eigvecs = np.linalg.eig(X)
    # print(eigvals)
    # # output = eigvecs @ np.diag(eigvals) @ eigvecs.T
    # # print(np.linalg.eigvals(output))
    # output = 0
    # for i in range(len(eigvals)):
    #     # print(eigvals[i])
    #     vi = eigvecs[:, i]
    #     # print(vi @ X @ vi)
    #     if eigvals[i] >= 0:
    #         output += eigvals[i] * np.outer(vi, vi)

    # print('out eigvals:', np.round(np.linalg.eigvals(output), 6))

    # # print('orig eigvals:', np.linalg.eigvals(X))
    # # U, S, VT = np.linalg.svd(X, full_matrices=True)
    # # S = np.maximum(S, 0)
    # # print('S:', S)
    # # # print(U.shape, S.shape, VT.shape)
    # # output = U @ np.diag(S) @ VT
    # # print(np.real(np.linalg.eigvals(output)))
    # # exit(0)

    # return output
    eigvals, eigvecs = np.linalg.eig(X)
    eigvals_clipped = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def ADMM(c, A, l, u, res=None):
    m, n = A.shape
    print(m, n)
    xk = np.zeros(n)
    zk = np.zeros(m)
    wk = np.zeros(n)
    yk = np.zeros(m)
    T = 5000
    sigma = .1
    rho = 1

    x_vals = [xk]
    z_vals = [zk]
    w_vals = [wk]
    y_vals = [yk]

    x_resids = []
    z_resids = []
    w_resids = []
    y_resids = []

    primal_resids = []
    dual_resids = []

    A = spa.csc_matrix(A)

    LHS = spa.bmat([
        [sigma * spa.eye(n), A.T],
        [A, -spa.eye(m) / rho]
    ])
    LHS = spa.csc_matrix(LHS)

    factor = spa.linalg.splu(LHS)
    # test = factor.solve(np.ones(157))

    for t in trange(T):
        rhs = np.hstack([sigma * xk - c - wk, zk - yk / rho])
        sol = factor.solve(rhs)
        xtilde_kplus1 = sol[:n]
        nu_kplus1 = sol[n:]
        # print(xtilde_kplus1.shape, nu_kplus1.shape)
        ztilde_kplus1 = zk + (nu_kplus1 - yk) / rho

        x_kplus1 = vec_psd_proj(xtilde_kplus1 + wk / sigma)
        z_kplus1 = box_proj(ztilde_kplus1 + yk / rho, l, u)

        w_kplus1 = wk + sigma * (xtilde_kplus1 - x_kplus1)
        y_kplus1 = yk + rho * (ztilde_kplus1 - z_kplus1)

        x_resids.append(np.linalg.norm(x_kplus1 - xk))
        z_resids.append(np.linalg.norm(z_kplus1 - zk))
        w_resids.append(np.linalg.norm(w_kplus1 - wk))
        y_resids.append(np.linalg.norm(y_kplus1 - yk))

        x_vals.append(xk)
        z_vals.append(zk)
        y_vals.append(yk)
        w_vals.append(wk)

        xk = x_kplus1
        zk = z_kplus1
        wk = w_kplus1
        yk = y_kplus1

        # print(np.linalg.norm(A @ xk - zk))
        # print(np.linalg.norm(c + A.T @ yk + wk))
        primal_resids.append(np.linalg.norm(A @ xk - zk))
        dual_resids.append(np.linalg.norm(c + A.T @ yk + wk))

    print(x_resids[-1], z_resids[-1], w_resids[-1], y_resids[-1])
    # print(c @ xk)
    obj_vals = [c @ x for x in x_vals]

    fig, (ax0, ax1) = plt.subplots(2, figsize=(6, 8))

    # fig.suptitle(f'CGAL progress, $K=1$, total time = {np.round(end-start, 3)} (s)')
    fig.suptitle('CGAL progress ADMM')
    ax0.plot(range(1, T+1), x_resids, label='X resid')
    ax0.plot(range(1, T+1), z_resids, label='z resid')
    ax0.plot(range(1, T+1), y_resids, label='y resid')
    ax0.plot(range(1, T+1), w_resids, label='w_resid')
    ax0.plot(range(1, T+1), primal_resids, label='primal_resid')
    ax0.plot(range(1, T+1), dual_resids, label='dual_resid')
    ax0.set_yscale('log')
    ax0.legend()

    ax1.plot(range(T+1), obj_vals, label='obj')
    if res is not None:
        ax1.axhline(y=res, linestyle='--', color='black')
    plt.xlabel('$t$')
    ax1.set_yscale('symlog')
    ax1.set_ylim(bottom=-.5)

    plt.savefig('Figure2.pdf')
    plt.show()


def main():
    np.random.seed(0)
    m = 5
    n = 3
    N = 1
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    # NNLS_cert_prob(n, m, A, N=N, t=.05)
    NNLS_test_cgal(n, m, A, N=N, t=.05)
    # GD_test(n, m, A, N=N, t=.05)
    # NNLS_test_cgal_combined(n, m, A, N=N, t=.05)


if __name__ == '__main__':
    main()
