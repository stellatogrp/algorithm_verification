import jax.numpy as jnp
from algocert.solvers.sdp_cgal_solver.cgal import cgal
import cvxpy as cp
import networkx as nx


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
    return C_op, A_op, A_star_op, b, alpha


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


def test_cgal_jax_maxcut():
    """
    tests lanczos jax implementation against the numpy implementation
        this is an accuracy test
        this is NOT a linear operator test: M is a matrix
    """
    n = 10
    m = n
    cgal_iters = 500

    # random Laplacian
    L = random_Laplacian_matrix(n)

    # problem data for cgal
    C_op, A_op, A_star_op, b, alpha = generate_maxcut_prob_data(L)

    # solve with cgal
    X, y, obj_vals, infeases, X_resids, y_resids = cgal(A_op, C_op, A_star_op, b, alpha, cgal_iters, m, n)

    # solve with cvxpy
    X_cvxpy = solve_maxcut_cvxpy(L)

    cgal_obj = jnp.trace(L @ X)
    cvxpy_obj = jnp.trace(L @ X_cvxpy)
    assert jnp.abs(cgal_obj - cvxpy_obj) / jnp.abs(cvxpy_obj) <= 1e-3
    assert jnp.linalg.norm(A_op(X) - b) <= 1e-2
