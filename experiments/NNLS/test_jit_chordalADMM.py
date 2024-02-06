#  import certification_problem.init_set as cpi
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa

# from algocert.basic_algorithm_steps.block_step import BlockStep
# from algocert.basic_algorithm_steps.linear_step import LinearStep
from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep

# from algocert.high_level_alg_steps.nonneg_lin_step import NonNegLinStep
from algocert.init_set.box_set import BoxSet

# from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.solvers.admm_chordal import chordal_solve, unvec_symm

# from algocert.solvers.sdp_cgal_solver.lanczos import approx_min_eigvec
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter

# from tqdm import trange


def NNLS_test(n, m, A, K=1, t=.05):
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

    solver_type = 'SDP_CGAL'
    CP = CertificationProblem(K, initsets, paramsets, obj, steps)
    # resg = CP.solve(solver_type=solver_type, add_bounds=True, add_RLT=False,
    #                 TimeLimit=3600, minimize=False, verbose=True)
    CP.canonicalize(solver_type=solver_type, add_bounds=True, add_RLT=False)
    h = CP.solver.handler
    # cp_res = h.test_with_cvxpy()
    # print(cp_res)

    # self.A_matrices = []
    # self.A_norms = []
    # self.b_lowerbounds = []
    # self.b_upperbounds = []
    # self.C_matrix = None
    # print(len(h.A_matrices))
    # print(h.b_upperbounds)

    # for A in h.A_matrices:
    #     print(type(A))
    A_list = [jnp.array(A.todense()) for A in h.A_matrices]
    l = jnp.array(h.b_lowerbounds)
    u = jnp.array(h.b_upperbounds)
    C = jnp.array(h.C_matrix.todense())
    dim = h.problem_dim
    print(dim)
    print(h.sample_iter_bound_map)
    chords = [jnp.arange(dim)]
    test_unscaled(A_list, C, l, u, chords, dim)
    # test_scaled(A_list, C, l, u, chords)


def test_unscaled(A_list, C, l, u, chords, n_orig):
    sol = chordal_solve(A_list, C, l, u, chords)
    z_k, iter_losses, z_all = sol
    nc2 = int(n_orig * (n_orig + 1) / 2)
    X_sol = unvec_symm(z_k[:nc2], n_orig)
    evals, evecs = jnp.linalg.eigh(X_sol)
    obj = jnp.trace(C @ X_sol)
    # print(iter_losses)
    print('obj:', obj)
    plt.plot(iter_losses, label='unscaled')
    plt.yscale('log')
    # plt.show()
    test_scaled(A_list, C, l, u, chords)


def test_scaled(A_list, C, l, u, chords):
    new_A = []
    new_l = []
    new_u = []
    # print(len(A_list), l.shape, u.shape)
    # exit(0)
    for i in range(len(A_list)):
        #     print(A_list[i])
        norm = jnp.linalg.norm(A_list[i], ord='fro')
        print(norm)
        new_A.append(A_list[i] / norm)
        new_l.append(l[i] / norm)
        new_u.append(u[i] / norm)
    new_l = jnp.array(new_l)
    new_u = jnp.array(new_u)
    sol = chordal_solve(new_A, C, new_l, new_u, chords)
    z_k, iter_losses, z_all = sol
    # print(iter_losses)
    plt.plot(iter_losses, linestyle='--', label='scaled')

    plt.legend()
    plt.yscale('log')
    plt.show()


def main():
    np.random.seed(0)
    m = 5
    n = 3
    K = 1
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    NNLS_test(n, m, A, K=K, t=.05)
    # NNLS_test_cgal(n, m, A, N=N, t=.05)
    # GD_test(n, m, A, N=N, t=.05)
    # NNLS_test_cgal_combined(n, m, A, N=N, t=.05)


if __name__ == '__main__':
    main()
