import jax.numpy as jnp
import numpy as np

from algoverify.solvers.sdp_cgal_solver.lanczos import (
    approx_min_eigvec,
    lanczos_matrix,
    scipy_mineigval,
)


def test_lanczos_numpy():
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


def test_lanczos_jax():
    """
    tests lanczos jax implementation against the numpy implementation
        this is an accuracy test
        this is NOT a linear operator test: M is a matrix
    """
    np.random.seed(0)
    n = 1000
    M = np.random.randn(n, n)
    M = (M + M.T) / 2
    num_lanczos_iters = 100

    # t0 = time.time()
    np_lambda, np_v = approx_min_eigvec(M, num_lanczos_iters, tol=1e-15)
    # np_time = time.time() - t0

    # t1 = time.time()
    jax_lambda, jax_v = lanczos_matrix(jnp.array(M), num_lanczos_iters)
    # jax_time = time.time() - t1

    # diff = Mv - lambda v
    #   should be close to zero for both methods
    diff_spa = (M @ np_v) - (jax_lambda * np_v)
    diff_lanczos = (M @ jax_v) - (jax_lambda * jax_v)

    # assert jax_time <= 0.5 * np_time
    assert np.linalg.norm(diff_spa) <= 1e-5
    assert np.linalg.norm(diff_lanczos) <= 1e-5
    assert np.abs(np_lambda - jax_lambda) <= 1e-5
    assert np.linalg.norm(np_v - jax_v) <= 1e-5 or np.linalg.norm(np_v + jax_v) <= 1e-5
