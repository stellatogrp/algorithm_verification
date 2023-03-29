import jax.numpy as jnp
import numpy as np

from algocert.solvers.sdp_cgal_solver.lanczos import (approx_min_eigvec,
                                                      lanczos_matrix,
                                                      scipy_mineigval)
from algocert.solvers.sdp_cgal_solver.cgal import cgal
import cvxpy as cp
import networkx as nx


def test_cgal_numpy():
    """
    tests lanczos numpy implementation against scipy minimum eigenvalue
        this is an accuracy test
        this is NOT a linear operator test: M is a matrix
    """
    np.random.seed(0)
    n = 100
    M = np.random.randn(n, n)
    M = (M + M.T) / 2
    num_lanczos_iters = 50

    spa_lambda, spa_v = scipy_mineigval(M)
    lanc_lambda, lanc_v = approx_min_eigvec(M, num_lanczos_iters)

    # diff = Mv - lambda v
    #   should be close to zero for both methods
    diff_spa = (M @ spa_v) - (spa_lambda * spa_v)
    diff_lanczos = (M @ lanc_v) - (lanc_lambda * lanc_v)

    assert np.linalg.norm(diff_spa) <= 1e-8
    assert np.linalg.norm(diff_lanczos) <= 1e-8
    assert np.abs(spa_lambda - lanc_lambda) <= 1e-8
    assert np.linalg.norm(spa_v - lanc_v) <= 1e-8 or np.linalg.norm(spa_v + lanc_v) <= 1e-8


def test_cgal_jax_maxcut():
    """
    tests lanczos jax implementation against the numpy implementation
        this is an accuracy test
        this is NOT a linear operator test: M is a matrix
    """
    n = 10
    m = n
    b = jnp.ones(n)
    alpha = n

    # create random Laplacian matrix for maxcut from Erdos-Renyi
    p = .5
    G = nx.erdos_renyi_graph(n, p)
    L_np = nx.linalg.laplacian_matrix(G).todense()
    L = jnp.array(L_np)

    # create operators
    def C_op(u):
        return -L @ u

    def A_op(X):
        return jnp.diag(X)

    def A_star_op(u, z):
        return jnp.multiply(u, z)

    cgal_iters = 500
    X, y, obj_vals, infeases, X_resids, y_resids = cgal(A_op, C_op, A_star_op, b, alpha, cgal_iters, m, n)

    # solve with cvxpy
    X_cvxpy = cp.Variable((n, n), symmetric=True)
    constraints = [X_cvxpy >> 0, cp.diag(X_cvxpy) == 1]
    obj = cp.Minimize(-cp.trace(L @ X_cvxpy))
    prob = cp.Problem(obj, constraints)
    prob.solve()

    cgal_obj = jnp.trace(L @ X)
    cvxpy_obj = jnp.trace(L @ X_cvxpy.value)
    assert jnp.abs(cgal_obj - cvxpy_obj) / jnp.abs(cvxpy_obj) <= 1e-3
    assert jnp.linalg.norm(A_op(X) - b) <= 1e-3
