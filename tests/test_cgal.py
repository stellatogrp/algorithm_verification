import jax.numpy as jnp
from algocert.solvers.sdp_cgal_solver.cgal import cgal, cgal_iteration, scale_problem_data, recover_original_sol, compute_frobenius_from_operator, compute_scale_factors, compute_operator_norm_from_A_vals
from experiments.NNLS.test_NNLS_ADMM import NNLS_test_cgal_copied
import cvxpy as cp
import networkx as nx
import time
from jax.experimental import sparse
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import os
import pytest
from scipy.sparse import csc_matrix


def partial_cgal_iter(A_op, C_op, A_star_op, b, beta, X, y, prev_v, lobpcg_iters, lobpcg_tol=1e-5):
    # m, n = y.size, X.shape[0]
    z = y + beta * (A_op(X) - b)
    A_star_partial_op = partial(A_star_op, z=jnp.expand_dims(z, 1))

    def evec_op(u):
        # we take the negative since lobpcg_standard finds the largest evec
        return -C_op(u) - A_star_partial_op(u)

    sol_out = sparse.linalg.lobpcg_standard(evec_op, prev_v, m=lobpcg_iters, tol=lobpcg_tol)
    eval_min, evec_min, iters = sol_out[0], sol_out[1], sol_out[2]

    return -eval_min, -evec_min, iters


def test_frobenius_norm_computation():
    """
    tests to ensure that the operator -> fro norm
        is done properly
    """
    m, n = 20, 45
    B = jnp.array(np.random.normal(size=(m, n)))
    def B_op(x):
        return B @ x
    
    B_op_F = compute_frobenius_from_operator(B_op, n)
    assert B_op_F == jnp.linalg.norm(B, 'fro')


def test_op_2_op_norm_computation():
    """
    tests to ensure that the operator -> op norm
        is done properly
    """
    m, n = 20, 45
    B = jnp.array(np.random.normal(size=(m, n, n)))
    

    # symmetrize
    for i in range(m):
        B = B.at[i, :, :].set((B[i, :, :] + B[i, :, :].T) / 2)

    B_stacked = jnp.zeros((m, n ** 2))
    for i in range(m):
        B_stacked = B_stacked.at[i, :].set(jnp.ravel(B[i, :, :]))

    op_norm_true = jnp.linalg.norm(B_stacked, ord=2)
    
    op_norm = compute_operator_norm_from_A_vals(B, m, n)
    
    assert op_norm == op_norm_true


def test_maxcut_scaling():
    n = 40
    m = n

    # random Laplacian
    L = random_Laplacian_matrix(n)

    # solve with cvxpy
    X_cvxpy = solve_maxcut_cvxpy(L)
    cvxpy_obj = -jnp.trace(L @ X_cvxpy)

    # problem data for cgal
    C_op, A_op, A_star_op, b, alpha, norm_A, scale_x_true, scale_c_true, scale_a_true = generate_maxcut_prob_data(L)
    scale_a, scale_c, scale_x = compute_scale_factors(C_op, A_op, alpha, m, n)

    # make sure the scale factors are accurate
    assert jnp.linalg.norm(scale_a - scale_a_true) <= 1e-10
    assert scale_c == scale_c_true
    assert scale_x == scale_x_true

    # make sure the resulting operators are the same
    scaled_data = scale_problem_data(C_op, A_op, A_star_op, alpha, norm_A, b, scale_x, scale_c, scale_a)
    C_op_scaled, A_op_scaled, A_star_op_scaled = scaled_data['C_op'], scaled_data['A_op'], scaled_data['A_star_op']
    alpha_scaled, norm_A_scaled, b_scaled = scaled_data['alpha'], scaled_data['norm_A'], scaled_data['b']
    rescale_obj, rescale_feas = scaled_data['rescale_obj'], scaled_data['rescale_feas']

    rand_vector = jnp.array(np.random.normal(size=(n)))
    rand_vector2 = jnp.array(np.random.normal(size=(n)))
    rand_matrix = jnp.array(np.random.normal(size=(n, n)))
    rand_matrix = (rand_matrix + rand_matrix.T) / 2
    # import pdb
    # pdb.set_trace()
    assert jnp.linalg.norm(C_op(rand_vector) - C_op_scaled(rand_vector) / scale_c) <= 1e-10
    assert jnp.linalg.norm(A_op(rand_matrix) - A_op_scaled(rand_matrix) / scale_a) <= 1e-10
    assert jnp.linalg.norm(A_star_op(rand_vector, rand_vector2) - 
                           A_star_op_scaled(rand_vector, rand_vector2) / scale_a) <= 1e-10


def random_Laplacian_matrix(n, p=.5):
    """
    create random Laplacian matrix for maxcut from Erdos-Renyi
    """
    G = nx.erdos_renyi_graph(n, p)
    L_np = nx.linalg.laplacian_matrix(G).todense()
    L = jnp.array(L_np)
    return L


def generate_maxcut_prob_data(L):
    """
    generates maxcut problem data from the Laplacian matrix
    """
    n = L.shape[0]

    # create operators
    def C_op(u):
        return -L @ u

    def A_op(X):
        return jnp.diag(X)

    def A_star_op(u, z):
        return jnp.multiply(u, z)
    b = jnp.ones(n)
    alpha = n
    norm_A = 1

    # data scaling
    scale_x = 1 / n
    scale_c = 1 / jnp.linalg.norm(L, ord='fro')
    scale_a = 1
    return C_op, A_op, A_star_op, b, alpha, norm_A, scale_x, scale_c, scale_a


def solve_maxcut_cvxpy(L):
    """
    solve maxcut with cvxpy with Laplacian matrix L
    """
    n = L.shape[0]
    X_cvxpy = cp.Variable((n, n), symmetric=True)
    constraints = [X_cvxpy >> 0, cp.diag(X_cvxpy) == 1]
    obj = cp.Minimize(-cp.trace(L @ X_cvxpy))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS)
    return jnp.array(X_cvxpy.value)


def solve_sdp(A_matrices, b_lower, b_upper, C, alpha):
    m = b_lower.size
    n = C.shape[0]
    X = cp.Variable((n, n), symmetric=True)

    constraints = [X >> 0]
    for i in range(m):
        if b_lower[i] > -np.inf:
            constraints.append(cp.trace(A_matrices[i] @ X) >= b_lower[i])
        if b_upper[i] < np.inf:
            constraints.append(cp.trace(A_matrices[i] @ X) <= b_upper[i])

    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve(solver=cp.MOSEK, verbose=True)
    return prob.value, X.value


@pytest.mark.skip(reason="temp")
def test_algocert():
    m_orig, n_orig = 3, 3
    A = csc_matrix(np.random.normal(size=(m_orig, n_orig)))
    cert_prob = NNLS_test_cgal_copied(n_orig, m_orig, A)

    handler = cert_prob.solver.handler

    C_jax = jnp.array(handler.C_matrix.todense())
    A_matrices = handler.A_matrices
    m = len(A_matrices)
    n = C_jax.shape[0]
    b = jnp.zeros((m, 2))
    b = b.at[:, 0].set(jnp.array(handler.b_lowerbounds))
    b = b.at[:, 1].set(jnp.array(handler.b_upperbounds))

    # scale As and b
    scaled_A_list, scaled_b = scale_matrices_b(handler.A_list, b, handler.A_norms)

    C_scaled = C_jax / C_jax

    A_matrices_jax_list = []
    for i in range(m):
        A_matrices_jax_list.append(jnp.array(A_matrices[i].todense()))

    def C_op(x):
        return C_jax @ x
    
    def A_op(X):
        Ax = jnp.zeros(m)
        for i in range(m):
            Ax = Ax.at[i].set(jnp.trace(A_matrices_jax_list[i] @ X))
        return Ax
    
    def A_star_op(u, z):
        """
        returns (A^* z)u 
            u has shape (n)
            z has shape (m)
        """
        zA = jnp.zeros((n, n))
        for i in range(m):
            zA += z[i] * A_matrices_jax_list[i]
        return zA @ u
    
    ###### solve with scs
    A_matrices_dense = [np.array(A.todense()) for A in A_matrices]
    scs_optval, X_scs = solve_sdp(A_matrices_dense, np.array(b[:, 0]), np.array(b[:, 1]), 
                                  np.array(handler.C_matrix.todense()), np.inf)
    
    ###### solve with cgal
    alpha = np.trace(X_scs)

    # norm_A = 1
    # cgal_iters = 100
    # rescale_obj_orig, rescale_feas_orig = 1, 1
    # cgal_scaled_out = cgal(A_op, C_op, A_star_op, b, alpha, norm_A,
    #                        rescale_obj_orig, rescale_feas_orig,
    #                        cgal_iters, m, n, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=True, jit=True)
    # X, y = cgal_scaled_out['X'], cgal_scaled_out['y']
    # obj_vals, infeases = cgal_scaled_out['obj_vals'], cgal_scaled_out['infeases']
    # X_resids, y_resids = cgal_scaled_out['X_resids'], cgal_scaled_out['y_resids']
    # lobpcg_steps = cgal_scaled_out['lobpcg_steps']

    import pdb
    pdb.set_trace()

    ###### solve with cgal scaled
    
    cgal_iters = 100
    # norm_A = 
    # scale_x = 
    # scale_a = 
    # scale_c = 
    scaled_data = scale_problem_data(C_op, A_op, A_star_op, alpha, norm_A, b, scale_x, scale_c, scale_a)
    C_op_scaled, A_op_scaled, A_star_op_scaled = scaled_data['C_op'], scaled_data['A_op'], scaled_data['A_star_op']
    alpha_scaled, norm_A_scaled, b_scaled = scaled_data['alpha'], scaled_data['norm_A'], scaled_data['b']
    rescale_obj, rescale_feas = scaled_data['rescale_obj'], scaled_data['rescale_feas']

    cgal_scaled_out = cgal(A_op_scaled, C_op_scaled, A_star_op_scaled, b_scaled, alpha_scaled, norm_A_scaled,
                           rescale_obj, rescale_feas,
                           cgal_iters, m, n, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=True, jit=True)
    X_scaled, y_scaled = cgal_scaled_out['X'], cgal_scaled_out['y']
    obj_vals_scaled, infeases_scaled = cgal_scaled_out['obj_vals'], cgal_scaled_out['infeases']
    X_resids_scaled, y_resids_scaled = cgal_scaled_out['X_resids'], cgal_scaled_out['y_resids']
    lobpcg_steps_scaled = cgal_scaled_out['lobpcg_steps']
    X_recovered, y_recovered = recover_original_sol(X_scaled, y_scaled, scale_x, scale_c, scale_a)

    # relative measures of success
    rel_obj_scaled = jnp.abs(cvxpy_obj - obj_vals_scaled) / (1 + jnp.abs(cvxpy_obj))
    rel_infeas_scaled = infeases_scaled / (1 + jnp.linalg.norm(b))

    


@pytest.mark.skip(reason="temp")
def test_warm_start_lobpcg():
    """
    warm starting lobpcg should not make any difference at all literally
    because
    this test checks this
    """
    n = 40
    m = n
    cgal_iters = 500

    # random Laplacian
    L = random_Laplacian_matrix(n)

    # solve with cvxpy
    X_cvxpy = solve_maxcut_cvxpy(L)
    cvxpy_obj = -jnp.trace(L @ X_cvxpy)

    # problem data for cgal
    C_op, A_op, A_star_op, b, alpha, norm_A, scale_x, scale_c, scale_a = generate_maxcut_prob_data(L)

    ###### solve with cgal
    rescale_obj_orig, rescale_feas_orig = 1, 1
    cgal_scaled_out = cgal(A_op, C_op, A_star_op, b, alpha, norm_A,
                           rescale_obj_orig, rescale_feas_orig,
                           cgal_iters, m, n, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=True, jit=True)
    obj_vals, infeases = cgal_scaled_out['obj_vals'], cgal_scaled_out['infeases']
    lobpcg_steps = cgal_scaled_out['lobpcg_steps']

    # relative measures of success
    rel_obj = jnp.abs(cvxpy_obj - obj_vals) / (1 + jnp.abs(cvxpy_obj))
    rel_infeas = infeases / (1 + jnp.linalg.norm(b))

    assert rel_obj[-1] <= 5e-2 and rel_obj[0] >= .05
    assert rel_infeas[-1] <= 5e-2 and rel_infeas[0] >= .1


    ###### solve with cgal with data scaling
    cgal_scaled_out_cold = cgal(A_op, C_op, A_star_op, b, alpha, norm_A,
                           rescale_obj_orig, rescale_feas_orig,
                           cgal_iters, m, n, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=False, jit=True)
    obj_vals_cold, infeases_cold = cgal_scaled_out['obj_vals'], cgal_scaled_out['infeases']
    lobpcg_steps_cold = cgal_scaled_out['lobpcg_steps']

    # relative measures of success
    rel_obj = jnp.abs(cvxpy_obj - obj_vals_cold) / (1 + jnp.abs(cvxpy_obj))
    rel_infeas = infeases_cold / (1 + jnp.linalg.norm(b))

    assert rel_obj[-1] <= 1e-2 and rel_obj[0] >= .05
    assert rel_infeas[-1] <= 1e-2 and rel_infeas[0] >= .1

    # compare lobpcg_iters -- they are identical
    assert jnp.linalg.norm(lobpcg_steps_cold - lobpcg_steps) == 0


@pytest.mark.skip(reason="temp")
def test_cgal_jit_speed():
    n = 100
    m = n
    cgal_iters = 20

    # random Laplacian
    L = random_Laplacian_matrix(n)

    # solve with cvxpy
    X_cvxpy = solve_maxcut_cvxpy(L)
    cvxpy_obj = -jnp.trace(L @ X_cvxpy)

    # problem data for cgal
    C_op, A_op, A_star_op, b, alpha, norm_A, scale_x, scale_c, scale_a = generate_maxcut_prob_data(L)

    ###### solve with jit
    t0_jit = time.time()
    rescale_obj_orig, rescale_feas_orig = 1, 1
    cgal_scaled_out = cgal(A_op, C_op, A_star_op, b, alpha, norm_A,
                           rescale_obj_orig, rescale_feas_orig,
                           cgal_iters, m, n, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=True, jit=True)
    X, y = cgal_scaled_out['X'], cgal_scaled_out['y']
    obj_vals, infeases = cgal_scaled_out['obj_vals'], cgal_scaled_out['infeases']
    X_resids, y_resids = cgal_scaled_out['X_resids'], cgal_scaled_out['y_resids']
    lobpcg_steps = cgal_scaled_out['lobpcg_steps']
    jit_time = time.time() - t0_jit

    # relative measures of success
    # rel_obj = jnp.abs(cvxpy_obj - obj_vals) / (1 + jnp.abs(cvxpy_obj))
    # rel_infeas = infeases / (1 + jnp.linalg.norm(b))

    # assert rel_obj[-1] <= 1e-3 and rel_obj[0] >= .05
    # assert rel_infeas[-1] <= 1e-2 and rel_infeas[0] >= .5

    ###### solve without jit
    t0_non_jit = time.time()
    rescale_obj_orig, rescale_feas_orig = 1, 1
    cgal_scaled_out = cgal(A_op, C_op, A_star_op, b, alpha, norm_A,
                           rescale_obj_orig, rescale_feas_orig,
                           cgal_iters, m, n, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=True, jit=False)
    X, y = cgal_scaled_out['X'], cgal_scaled_out['y']
    obj_vals, infeases = cgal_scaled_out['obj_vals'], cgal_scaled_out['infeases']
    X_resids, y_resids = cgal_scaled_out['X_resids'], cgal_scaled_out['y_resids']
    lobpcg_steps = cgal_scaled_out['lobpcg_steps']
    non_jit_time = time.time() - t0_non_jit

    # jitting should reduce time by at least 90%
    assert jit_time <= .1 * non_jit_time


@pytest.mark.skip(reason="temp")
def test_cgal_scaling_maxcut():
    n = 100
    m = n
    cgal_iters = 1000

    # random Laplacian
    L = random_Laplacian_matrix(n)

    # solve with cvxpy
    X_cvxpy = solve_maxcut_cvxpy(L)
    cvxpy_obj = -jnp.trace(L @ X_cvxpy)

    # problem data for cgal
    C_op, A_op, A_star_op, b, alpha, norm_A, scale_x, scale_c, scale_a = generate_maxcut_prob_data(L)

    ###### solve with cgal
    rescale_obj_orig, rescale_feas_orig = 1, 1
    cgal_scaled_out = cgal(A_op, C_op, A_star_op, b, alpha, norm_A,
                           rescale_obj_orig, rescale_feas_orig,
                           cgal_iters, m, n, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=True, jit=True)
    X, y = cgal_scaled_out['X'], cgal_scaled_out['y']
    obj_vals, infeases = cgal_scaled_out['obj_vals'], cgal_scaled_out['infeases']
    X_resids, y_resids = cgal_scaled_out['X_resids'], cgal_scaled_out['y_resids']
    lobpcg_steps = cgal_scaled_out['lobpcg_steps']

    # relative measures of success
    rel_obj = jnp.abs(cvxpy_obj - obj_vals) / (1 + jnp.abs(cvxpy_obj))
    rel_infeas = infeases / (1 + jnp.linalg.norm(b))

    assert rel_obj[-1] <= 1e-3 and rel_obj[0] >= .05
    assert rel_infeas[-1] <= 1e-2 and rel_infeas[0] >= .1


    ###### solve with cgal with data scaling
    scaled_data = scale_problem_data(C_op, A_op, A_star_op, alpha, norm_A, b, scale_x, scale_c, scale_a)
    C_op_scaled, A_op_scaled, A_star_op_scaled = scaled_data['C_op'], scaled_data['A_op'], scaled_data['A_star_op']
    alpha_scaled, norm_A_scaled, b_scaled = scaled_data['alpha'], scaled_data['norm_A'], scaled_data['b']
    rescale_obj, rescale_feas = scaled_data['rescale_obj'], scaled_data['rescale_feas']

    cgal_scaled_out = cgal(A_op_scaled, C_op_scaled, A_star_op_scaled, b_scaled, alpha_scaled, norm_A_scaled,
                           rescale_obj, rescale_feas,
                           cgal_iters, m, n, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=True, jit=True)
    X_scaled, y_scaled = cgal_scaled_out['X'], cgal_scaled_out['y']
    obj_vals_scaled, infeases_scaled = cgal_scaled_out['obj_vals'], cgal_scaled_out['infeases']
    X_resids_scaled, y_resids_scaled = cgal_scaled_out['X_resids'], cgal_scaled_out['y_resids']
    lobpcg_steps_scaled = cgal_scaled_out['lobpcg_steps']
    X_recovered, y_recovered = recover_original_sol(X_scaled, y_scaled, scale_x, scale_c, scale_a)

    # relative measures of success
    rel_obj_scaled = jnp.abs(cvxpy_obj - obj_vals_scaled) / (1 + jnp.abs(cvxpy_obj))
    rel_infeas_scaled = infeases_scaled / (1 + jnp.linalg.norm(b))

    assert rel_obj_scaled[-1] <= 1e-2 and rel_obj_scaled[0] >= .05
    assert rel_infeas_scaled[-1] <= 1e-2 and rel_infeas_scaled[0] >= .5

    # create plots
    os.mkdir('outputs')
    os.mkdir('outputs/test_results')
    dir = 'outputs/test_results'
    plt.plot(rel_obj, label='unscaled rel obj')
    plt.plot(rel_obj_scaled, label='scaled rel obj')
    plt.plot(rel_infeas, label='unscaled rel infeas')
    plt.plot(rel_infeas_scaled, label='scaled infeas')
    plt.yscale('log')
    plt.xlabel('cgal iterations')
    plt.legend()
    plt.savefig(f"{dir}/maxcut_obj_infeas.pdf")
    plt.clf()

    plt.plot(X_resids, label='unscaled X_resids')
    plt.plot(X_resids_scaled, label='scaled X_resids')
    plt.plot(y_resids, label='unscaled y_resids')
    plt.plot(y_resids_scaled, label='scaled y_resids')
    plt.xlabel('cgal iterations')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"{dir}/maxcut_resids.pdf")




# def test_cgal_maxcut():
#     """
#     tests lanczos jax implementation against the numpy implementation
#         this is an accuracy test
#         this is NOT a linear operator test: M is a matrix
#     """
#     n = 100
#     m = n
#     cgal_iters = 100

#     # random Laplacian
#     L = random_Laplacian_matrix(n)

#     # problem data for cgal
#     C_op, A_op, A_star_op, b, alpha, scale_x, scale_c, scale_a = generate_maxcut_prob_data(L)

#     # solve with cvxpy
#     X_cvxpy = solve_maxcut_cvxpy(L)

#     # solve with cgal
#     cgal_out = cgal(A_op, C_op, A_star_op, b, alpha, cgal_iters, m, n,
#                     lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=False)
#     X, y, obj_vals, infeases, X_resids, y_resids = cgal_out

#     cgal_obj = jnp.trace(L @ X)
#     cvxpy_obj = jnp.trace(L @ X_cvxpy)

#     import pdb
#     pdb.set_trace()

#     assert jnp.abs(cgal_obj - cvxpy_obj) / jnp.abs(cvxpy_obj) <= 1e-3
#     assert jnp.linalg.norm(A_op(X) - b) <= 1e-2


# def test_cgal_warmstart_v_maxcut():
#     """
#     for maxcut
#     tests that warm-starting lobpcg is faster than cold-starting lobpcg
#     """
#     n = 10
#     m = n
#     cgal_iters = 10

#     # random Laplacian
#     L = random_Laplacian_matrix(n)

#     # problem data for cgal
#     C_op, A_op, A_star_op, b, alpha = generate_maxcut_prob_data(L)

#     # solve with cgal cold start eigensolver
#     t0_cs = time.time()
#     out_cold_start = cgal(A_op, C_op, A_star_op, b, alpha, cgal_iters, m, n,
#                           warm_start_v=False)
#     X_cold_start = out_cold_start[0]
#     time_cs = time.time() - t0_cs

#     # solve with cgal warm start eigensolver
#     t0_ws = time.time()
#     out_warm_start = cgal(A_op, C_op, A_star_op, b, alpha, cgal_iters, m, n,
#                           warm_start_v=True)
#     X_warm_start = out_warm_start[0]
#     time_ws = time.time() - t0_ws

#     # assert time_ws < .9 * time_cs
#     # solve with cvxpy
#     X_cvxpy = solve_maxcut_cvxpy(L)
#     cvxpy_obj = jnp.trace(L @ X_cvxpy)

#     cs_inf = jnp.linalg.norm(A_op(X_cold_start) - b)
#     cs_cgal_obj = jnp.trace(L @ X_cold_start)
#     cs_obj_diff = jnp.abs(cs_cgal_obj - cvxpy_obj) / jnp.abs(cvxpy_obj)
    
#     ws_inf = jnp.linalg.norm(A_op(X_warm_start) - b)
#     ws_cgal_obj = jnp.trace(L @ X_cold_start)
#     ws_obj_diff = jnp.abs(ws_cgal_obj - cvxpy_obj) / jnp.abs(cvxpy_obj)

#     # assert that warm-starting is better
#     assert ws_inf < cs_inf
#     assert ws_obj_diff <= cs_obj_diff

    

#     # cgal_obj = jnp.trace(L @ X_cold_start)
#     # cvxpy_obj = jnp.trace(L @ X_cvxpy)
#     # assert jnp.abs(cgal_obj - cvxpy_obj) / jnp.abs(cvxpy_obj) <= 1e-3
#     # assert jnp.linalg.norm(A_op(X_cold_start) - b) <= 1e-2

#     # cgal_obj = jnp.trace(L @ X_warm_start)
#     # cvxpy_obj = jnp.trace(L @ X_cvxpy)
#     # assert jnp.abs(cgal_obj - cvxpy_obj) / jnp.abs(cvxpy_obj) <= 1e-3
#     # assert jnp.linalg.norm(A_op(X_warm_start) - b) <= 1e-2
