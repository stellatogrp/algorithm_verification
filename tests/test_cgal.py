import jax.numpy as jnp
from algocert.solvers.sdp_cgal_solver.cgal import cgal, cgal_iteration, scale_problem_data, recover_original_sol, compute_frobenius_from_operator, compute_scale_factors, compute_operator_norm_from_A_vals, relative_infeas, relative_obj
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


@pytest.mark.skip(reason="temp")
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

    assert jnp.linalg.norm(C_op(rand_vector) - C_op_scaled(rand_vector) / scale_c) <= 1e-10
    assert jnp.linalg.norm(A_op(rand_matrix) - A_op_scaled(rand_matrix) / scale_a) <= 1e-10
    assert jnp.linalg.norm(A_star_op(rand_vector, rand_vector2) - 
                           A_star_op_scaled(rand_vector, rand_vector2) / scale_a) <= 1e-10
    assert jnp.abs(alpha_scaled - 1) <= 1e-10
    assert jnp.linalg.norm(b - b_scaled / (scale_a * scale_x)) <= 1e-10


def random_Laplacian_matrix(n, p=.5):
    """
    create random Laplacian matrix for maxcut from Erdos-Renyi
    """
    G = nx.erdos_renyi_graph(n, p)
    L_np = nx.linalg.laplacian_matrix(G).todense()
    L = jnp.array(L_np)
    return L


def generate_random_phase_retrieval(m, n):
    # A_matrices_np = jnp.zeros((m, n, n))
    C = np.eye(n)

    # generate random a_i i=1,...,m vectors
    a_signals = jnp.array(np.random.normal(size=(m, n)))
    A_data = jnp.zeros((m, n, n))
    for i in range(m):
        A_data = A_data.at[i, :, :].set(jnp.outer(a_signals[i, :], a_signals[i, :]))

    # generate true x
    x = jnp.array(np.random.normal(size=(n)))
    X = jnp.outer(x, x)

    # generate b from x
    b = jnp.zeros(m)
    for i in range(m):
        b = b.at[i].set(jnp.trace(A_data[i, :, :] @ X))

    return A_data, b, C


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


def solve_generic_cvxpy(A_data, b, C):
    """
    solve maxcut with cvxpy with Laplacian matrix L
    """
    n = C.shape[0]

    assert b.ndim == 1 or b.ndim == 2
    
    if b.ndim == 1:
        m = b.size
        b_case = 'equality'
    elif b.ndim == 2:
        m = b.shape[0]
        b_case = 'inequality'
        assert b.shape[1] == 2

    X = cp.Variable((n, n), symmetric=True)
    constraints = [X >> 0]
    
    for i in range(m):
        if b_case == 'equality':
            constraints.append(cp.trace(A_data[i, :, :] @ X) == b[i])
        elif b_case == 'inequality':
            if b[i, 0] > -np.inf:
                constraints.append(cp.trace(A_data[i, :, :] @ X) >= b[i, 0])
            if b[i, 1] < np.inf:
                constraints.append(cp.trace(A_data[i, :, :] @ X) <= b[i, 1])
        
    obj = cp.Minimize(cp.trace(C @ X))
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS)
    return jnp.array(X.value), prob.value


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
    # rel_obj_scaled = jnp.abs(cvxpy_obj - obj_vals_scaled) / (1 + jnp.abs(cvxpy_obj))
    # rel_infeas_scaled = infeases_scaled / (1 + jnp.linalg.norm(b))
    rel_obj_scaled = relative_obj(obj_vals_scaled, cvxpy_obj)
    rel_infeas_scaled = relative_infeas(infeases_scaled, b)


@pytest.mark.skip(reason="temp")  
def test_warm_start_lobpcg():
    """
    warm starting lobpcg should not make any difference at all literally
    because
    this test checks this
    """
    n = 30
    m = n
    cgal_iters = 30

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

    # assert rel_obj[-1] <= 5e-2 and rel_obj[0] >= .05
    # assert rel_infeas[-1] <= 5e-2 and rel_infeas[0] >= .1


    ###### solve with cgal with data scaling
    cgal_scaled_out_cold = cgal(A_op, C_op, A_star_op, b, alpha, norm_A,
                           rescale_obj_orig, rescale_feas_orig,
                           cgal_iters, m, n, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=False, jit=True)
    obj_vals_cold, infeases_cold = cgal_scaled_out['obj_vals'], cgal_scaled_out['infeases']
    lobpcg_steps_cold = cgal_scaled_out['lobpcg_steps']

    # relative measures of success
    # rel_obj = jnp.abs(cvxpy_obj - obj_vals_cold) / (1 + jnp.abs(cvxpy_obj))
    # rel_infeas = infeases_cold / (1 + jnp.linalg.norm(b))
    rel_obj = relative_obj(obj_vals_cold, cvxpy_obj)
    rel_infeas = relative_infeas(infeases_cold, b)

    # assert rel_obj[-1] <= 1e-2 and rel_obj[0] >= .05
    # assert rel_infeas[-1] <= 1e-2 and rel_infeas[0] >= .1

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


def generate_primitives_from_matrices(A_data, C):
    m, n, _ = A_data.shape
    def C_op(x):
        return C @ x
    
    def A_op(X):
        AX = jnp.zeros(m)
        for i in range(m):
            AX = AX.at[i].set(jnp.trace(A_data[i, :, :] @ X))
        return AX
    
    def A_star_op(u, z):
        zA = jnp.zeros((n, n))
        for i in range(m):
            zA += z[i] * A_data[i, :, :]
        return zA @ u
        
    return C_op, A_op, A_star_op


def test_phase_retrieval_scaling():
    # generate random A, b, C
    m, n = 150, 50   
    A_data, b, C = generate_random_phase_retrieval(m, n)

    # solve with cvxpy
    X_opt, optval = solve_generic_cvxpy(A_data, b, C)
    alpha_star = jnp.trace(X_opt)

    # generate problem data for cgal -- 3 primitives, b
    #   use alpha from cvxpy
    C_op, A_op, A_star_op = generate_primitives_from_matrices(A_data, C)
    A_data_stacked = jnp.reshape(A_data, (m, n ** 2))
    norm_A = jnp.linalg.norm(A_data_stacked, ord=2)

    # scale problem data

    


    # solve original with cgal
    cgal_iters = 100
    cgal_out = cgal(A_op, C_op, A_star_op, b, 
                    alpha=2 * alpha_star, norm_A=norm_A, 
                    rescale_obj=1, rescale_feas=1, 
                    cgal_iters=cgal_iters, m=m, n=n, beta0=1, jit=True)
    obj_vals, infeases = cgal_out['obj_vals'], cgal_out['infeases']
    X_resids, y_resids = cgal_out['X_resids'], cgal_out['y_resids']
    rel_objs = relative_obj(obj_vals, optval)
    rel_infeases = relative_infeas(infeases, b)

    # solve scaled with cgal
    cgal_out_scaled, X_recovered, y_recovered, scaled_data, scale_factors = scale_and_solve(C_op, A_op, A_star_op, 
                                                         2 * alpha_star, norm_A, b, m, n, cgal_iters, beta0=1, jit=True
                                                         )
    obj_vals_scaled, infeases_scaled = cgal_out_scaled['obj_vals'], cgal_out_scaled['infeases']
    rel_objs_scaled = relative_obj(obj_vals_scaled, optval)
    rel_infeases_scaled = relative_infeas(infeases_scaled, b)
    X_resids_scaled, y_resids_scaled = cgal_out_scaled['X_resids'], cgal_out_scaled['y_resids']

    # test the scaling
    rand_vector_n = jnp.array(np.random.normal(size=(n)))
    rand_vector_m = jnp.array(np.random.normal(size=(m, 1)))
    rand_matrix = jnp.array(np.random.normal(size=(n, n)))
    rand_matrix = (rand_matrix + rand_matrix.T) / 2
    
    scale_a = scale_factors['scale_a']
    scale_c = scale_factors['scale_c']
    scale_x = scale_factors['scale_x']
    A_op_scaled = scaled_data['A_op']
    A_star_op_scaled = scaled_data['A_star_op']
    C_op_scaled = scaled_data['C_op']
    alpha_scaled = scaled_data['alpha']

    # check the operators are truly scaled
    assert jnp.linalg.norm(A_op_scaled(rand_matrix) - jnp.multiply(A_op(rand_matrix), scale_a)) <= 1e-6
    assert jnp.linalg.norm(C_op_scaled(rand_vector_n) - jnp.multiply(C_op(rand_vector_n), scale_c)) <= 1e-6
    A_star_scaled_val = A_star_op_scaled(rand_matrix, rand_vector_m)
    scaled_rand_vector_m = jnp.multiply(rand_vector_m, jnp.expand_dims(scale_a, 1))
    A_star_val = A_star_op(rand_matrix, scaled_rand_vector_m)
    assert jnp.linalg.norm(A_star_scaled_val - A_star_val) <= 1e-6
    assert jnp.abs(alpha_scaled - 1) <= 1e-8

    create_plots('phase_retrieval', rel_objs, rel_objs_scaled, rel_infeases, rel_infeases_scaled, 
                 X_resids, X_resids_scaled, y_resids, y_resids_scaled)

    import pdb
    pdb.set_trace()
    assert False

    # check the frobenius norms of each A_i


    # check the operator norm of A_stacked


    
    # create plots


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
    # scaled_data = scale_problem_data(C_op, A_op, A_star_op, alpha, norm_A, b, scale_x, scale_c, scale_a)
    # C_op_scaled, A_op_scaled, A_star_op_scaled = scaled_data['C_op'], scaled_data['A_op'], scaled_data['A_star_op']
    # alpha_scaled, norm_A_scaled, b_scaled = scaled_data['alpha'], scaled_data['norm_A'], scaled_data['b']
    # rescale_obj, rescale_feas = scaled_data['rescale_obj'], scaled_data['rescale_feas']

    # cgal_scaled_out = cgal(A_op_scaled, C_op_scaled, A_star_op_scaled, b_scaled, alpha_scaled, norm_A_scaled,
    #                        rescale_obj, rescale_feas,
    #                        cgal_iters, m, n, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=True, jit=True)
    cgal_scaled_out, X_recovered, y_recovered = scale_and_solve(C_op, A_op, A_star_op, alpha, 
                                                                norm_A, b, scale_x, scale_c, 
                                                                scale_a, m, n, cgal_iters)
    obj_vals_scaled, infeases_scaled = cgal_scaled_out['obj_vals'], cgal_scaled_out['infeases']
    X_resids_scaled, y_resids_scaled = cgal_scaled_out['X_resids'], cgal_scaled_out['y_resids']

    # relative measures of success
    rel_obj_scaled = jnp.abs(cvxpy_obj - obj_vals_scaled) / (1 + jnp.abs(cvxpy_obj))
    rel_infeas_scaled = infeases_scaled / (1 + jnp.linalg.norm(b))

    assert rel_obj_scaled[-1] <= 1e-2 and rel_obj_scaled[0] >= .05
    assert rel_infeas_scaled[-1] <= 1e-2 and rel_infeas_scaled[0] >= .5

    create_plots('maxcut', rel_obj, rel_obj_scaled, rel_infeas, rel_infeas_scaled, 
                 X_resids, X_resids_scaled, y_resids, y_resids_scaled)
    

def scale_and_solve(C_op, A_op, A_star_op, alpha, norm_A, b, m, n, cgal_iters, beta0=1, jit=True):
    # find scale factors
    scale_a, scale_c, scale_x = compute_scale_factors(C_op, A_op, alpha, m, n)

    # scale problem data
    scaled_data = scale_problem_data(C_op, A_op, A_star_op, alpha, norm_A, b, scale_x, scale_c, scale_a)
    C_op_scaled, A_op_scaled, A_star_op_scaled = scaled_data['C_op'], scaled_data['A_op'], scaled_data['A_star_op']
    alpha_scaled, norm_A_scaled, b_scaled = scaled_data['alpha'], scaled_data['norm_A'], scaled_data['b']
    rescale_obj, rescale_feas = scaled_data['rescale_obj'], scaled_data['rescale_feas']

    # x0 = jnp.ones(n)
    # import pdb
    # pdb.set_trace()

    # run cgal
    cgal_scaled_out = cgal(A_op_scaled, C_op_scaled, A_star_op_scaled, b_scaled, alpha_scaled, norm_A_scaled,
                           rescale_obj, rescale_feas,
                           cgal_iters, m, n, beta0=beta0, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=True, jit=jit)
    X_scaled, y_scaled = cgal_scaled_out['X'], cgal_scaled_out['y']
    X_recovered, y_recovered = recover_original_sol(X_scaled, y_scaled, scale_x, scale_c, scale_a)

    scale_factors = dict(scale_a=scale_a, scale_c=scale_c, scale_x=scale_x)
    return cgal_scaled_out, X_recovered, y_recovered, scaled_data, scale_factors


def create_plots(title, rel_obj, rel_obj_scaled, rel_infeas, rel_infeas_scaled, 
                 X_resids, X_resids_scaled, y_resids, y_resids_scaled, ylim=(1e-5,1)):
    """
    given a title (e.g. maxcut or phase_retrieval) and a suite of measures of cgal's progress
        this creates plots

    this function is specialized for 
    """
    # create plots
    dir = 'outputs/test_results'
    if not os.path.exists('outputs'):
        os.mkdir('outputs')
    if not os.path.exists(dir):
        os.mkdir(dir)
    
    plt.plot(rel_obj, label='unscaled rel obj')
    plt.plot(rel_obj_scaled, label='scaled rel obj')
    plt.plot(rel_infeas, label='unscaled rel infeas')
    plt.plot(rel_infeas_scaled, label='scaled infeas')
    plt.ylim(ylim)
    plt.yscale('log')
    plt.xlabel('cgal iterations')
    plt.legend()
    plt.savefig(f"{dir}/{title}_obj_infeas.pdf")
    plt.clf()

    plt.plot(X_resids, label='unscaled X_resids')
    plt.plot(X_resids_scaled, label='scaled X_resids')
    plt.plot(y_resids, label='unscaled y_resids')
    plt.plot(y_resids_scaled, label='scaled y_resids')
    # plt.ylim(ylim)
    plt.xlabel('cgal iterations')
    plt.yscale('log')
    plt.legend()
    plt.savefig(f"{dir}/{title}_resids.pdf")




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
