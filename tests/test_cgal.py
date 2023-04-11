import jax.numpy as jnp
from algocert.solvers.sdp_cgal_solver.cgal import cgal, cgal_iteration, scale_problem_data, recover_original_sol
import cvxpy as cp
import networkx as nx
import time
from jax.experimental import sparse
from functools import partial
import numpy as np
import matplotlib.pyplot as plt


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


def test_lobpcg():
    n = 30
    m = n
    # cgal_iters = 1000

    # random Laplacian
    L = random_Laplacian_matrix(n)

    # problem data for cgal
    C_op, A_op, A_star_op, b, alpha, scale_x, scale_c, scale_a = generate_maxcut_prob_data(L)

    # init
    X = jnp.zeros((n, n))
    y = jnp.zeros(m)
    beta = 1

    z = y + beta * (A_op(X) - b)
    A_star_partial_op = partial(A_star_op, z=jnp.expand_dims(z, 1))

    def evec_op(u):
        # we take the negative since lobpcg_standard finds the largest evec
        return -C_op(u) - A_star_partial_op(u)

    # first cgal iteration
    iters = np.array([5, 10, 30, 90, 100])
    num_passes = iters.size
    errors = jnp.zeros(num_passes)
    num_steps = jnp.zeros(num_passes)
    for i in range(num_passes):
        lobpcg_out = partial_cgal_iter(A_op, C_op, A_star_op, b, beta, X, y, prev_v=jnp.zeros((n, 1)),
                                       lobpcg_iters=iters[i], lobpcg_tol=1e-15)
        lambd, v, curr_steps = lobpcg_out
        error = jnp.linalg.norm(evec_op(v) + lambd * v)
        errors = errors.at[i].set(error)
        num_steps = num_steps.at[i].set(curr_steps)
    assert jnp.all(jnp.diff(errors[:num_passes - 1]) < 0)
    assert jnp.all(num_steps[:3] == iters[:3])


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
    prob.solve()
    return jnp.array(X_cvxpy.value)


def test_cgal_scaling_maxcut():
    n = 100
    m = n
    cgal_iters = 500

    # random Laplacian
    L = random_Laplacian_matrix(n)

    # solve with cvxpy
    X_cvxpy = solve_maxcut_cvxpy(L)
    cvxpy_obj = jnp.trace(L @ X_cvxpy)

    # problem data for cgal
    C_op, A_op, A_star_op, b, alpha, norm_A, scale_x, scale_c, scale_a = generate_maxcut_prob_data(L)

    # solve with cgal
    # X, y, obj_vals, infeases, X_resids, y_resids = cgal(A_op, C_op, A_star_op, b, alpha, cgal_iters, m, n,
    #                                                     lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=False,
    #                                                     jit=True)

    # solve with cgal with data scaling
    scaled_data = scale_problem_data(C_op, A_op, A_star_op, alpha, norm_A, b, scale_x, scale_c, scale_a)
    C_op_scaled, A_op_scaled, A_star_op_scaled, alpha_scaled, norm_A_scaled, b_scaled = scaled_data

    cgal_scaled_out = cgal(A_op_scaled, C_op_scaled, A_star_op_scaled, b_scaled, alpha_scaled, norm_A_scaled,
                           cgal_iters, m, n, lobpcg_iters=1000, lobpcg_tol=1e-10, warm_start_v=True, jit=True)
    X_scaled, y_scaled = cgal_scaled_out['X'], cgal_scaled_out['y']
    obj_vals_scaled, infeases_scaled = cgal_scaled_out['obj_vals'], cgal_scaled_out['infeases']
    X_resids_scaled, y_resids_scaled = cgal_scaled_out['X_resids'], cgal_scaled_out['y_resids']
    lobpcg_steps_scaled = cgal_scaled_out['lobpcg_steps']
    X_recovered, y_recovered = recover_original_sol(X_scaled, y_scaled, scale_x, scale_c, scale_a)
    cgal_obj_scaled = jnp.trace(L @ X_recovered)
    infeas_scaled = jnp.linalg.norm(A_op(X_recovered) - b)

    import pdb
    pdb.set_trace()
    assert False


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
