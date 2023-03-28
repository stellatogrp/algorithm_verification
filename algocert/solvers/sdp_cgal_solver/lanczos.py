from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import scipy.sparse as spa
from jax import lax
from jax.config import config
from scipy.linalg import eigh_tridiagonal

config.update("jax_enable_x64", True)


def lanczos_matrix(M, lanczos_iters):
    n, n_ = M.shape

    def M_op(v):
        return M @ v
    return lanczos(M_op, lanczos_iters, n)


def lanczos(M_op, lanczos_iters, n):
    """
    Runs the lanczos method in jax to find (approx) minimum eigenvalue of
        matrix defined by M

    Inputs:
        M: linear operator that is a callable that returns M_op(x) = Mx
        lanczos_iters: number of iterations of the lanczos method
        n: M has shape (n, n)
            we pass it in to deal with the case of M being a linear operator

        this method jaxifies the approx_min_eigvec function

    Outputs:
        min_eval: minimum eigenvalue of M
        min_evec: corresponding eigenvector of the minimum eigenvalue of M

    In the end, we should have
        M(min_evec) = min_eval * min_evec
    """
    v_init = init_lanczos(n)

    v_mat, alpha_vec, beta_vec = lanczos_for_loop(M_op, v_init, lanczos_iters)

    min_eval, min_evec = finalize_lanczos(v_mat, alpha_vec, beta_vec)

    return min_eval, min_evec


def init_lanczos(n):
    key = jax.random.PRNGKey(0)
    v = jax.random.normal(key, (n,))
    return v / jnp.linalg.norm(v)


def finalize_lanczos(v_mat, alpha_vec, beta_vec):
    # beta_trunc needs to take first i-1 entries, while alpha_vec takes first i entries
    beta_trunc = beta_vec[:-1]

    # this line errors - todo figure out why
    # l_test, v_test = jax.scipy.linalg.eigh_tridiagonal(alpha_vec, beta_trunc, select='i', select_range=[0, 0])
    l_test, v_test = eigh_tridiagonal(alpha_vec, beta_trunc, select='i', select_range=[0, 0])

    xi = l_test[0]
    v = v_mat.T @ v_test
    v = v / jnp.linalg.norm(v)
    min_eval, min_evec = xi, v

    return min_eval, min_evec


def lanczos_for_loop(M_op, v_init, lanczos_iters):
    """
    calls lanczos_iter and uses the jax.fori_loop syntax
    val = jax.lax.fori_loop(0, iters, body_fn, val)

    recall the jax fori_loop looks like
    def fori_loop(lower, upper, body_fun, init_val):
        val = init_val
        for i in range(lower, upper):
            val = body_fun(i, val)
        return val
    """
    n = v_init.size
    lanczos_iter_partial = partial(lanczos_iter_fori_loop, M_op=M_op)
    raw_iter_partial = partial(lanczos_iter_raw, M_op=M_op)

    # initialize v_mat, alpha_vec, beta_vec
    v_mat = jnp.zeros((lanczos_iters, n))
    v_mat = v_mat.at[0, :].set(v_init)
    alpha_vec, beta_vec = jnp.zeros(lanczos_iters), jnp.zeros(lanczos_iters)

    # unroll the first iteration since the logic is different
    v_next, alpha, beta = raw_iter_partial(v_init, 0 * v_init, 0)
    v_mat = v_mat.at[1, :].set(v_next)
    alpha_vec = alpha_vec.at[0].set(alpha)
    beta_vec = beta_vec.at[0].set(beta)

    # do the for loop for the other (lanczos_iters - 2) iterations
    init_val = v_mat, alpha_vec, beta_vec
    val = lax.fori_loop(1, lanczos_iters - 1, lanczos_iter_partial, init_val)
    v_mat, alpha_vec, beta_vec = val

    # do the final iteration -- we do NOT care about the final v value
    #   the point is to fill in the last entry of alpha_vec and beta_vec
    v_curr, v_prev = v_mat[lanczos_iters - 1, :], v_mat[lanczos_iters - 2, :]
    beta_prev = beta_vec[lanczos_iters - 2]
    v_final, alpha_final, beta_final = raw_iter_partial(v_curr, v_prev, beta_prev)
    alpha_vec = alpha_vec.at[-1].set(alpha_final)
    beta_vec = beta_vec.at[-1].set(beta_final)

    return v_mat, alpha_vec, beta_vec


def lanczos_iter_fori_loop(i, val, M_op):
    """
    this functions calls lanczos_iter_raw
    serves as the body_fn for jax.lax.fori_loop
    """
    # setup
    v_mat, alpha_vec, beta_vec = val
    v_curr = v_mat[i, :]
    v_prev = v_mat[i - 1, :]
    beta_prev = beta_vec[i - 1]

    # iteration
    v_next, alpha, beta = lanczos_iter_raw(v_curr, v_prev, beta_prev, M_op)

    # set the entries
    alpha_vec = alpha_vec.at[i].set(alpha)
    beta_vec = beta_vec.at[i].set(beta)
    v_mat = v_mat.at[i + 1, :].set(v_next)

    return v_mat, alpha_vec, beta_vec


# @partial(jit, static_argnums=(3))
def lanczos_iter_raw(v_curr, v_prev, beta_prev, M_op):
    """
    runs an iteration of the lanczos algorithm

    alpha_i = Re(v_i*(M_op(v_i)))
    v_{i+1} = Mv_i - alpha_i v_i - beta_{i-1} v_{i-1}
    beta_i = ||v_i||_2
    if beta_i == 0 break: found an invariant subspace
    v_{i+1} = v_{i+1} / beta_i

    returns: v_{i+1}, v_i, alpha_i, beta_i

    we do NOT do storage optimality, where only v_{i+1} and v_i are stored

    details for indices
        at index i, we set
            v_{i + 1}, alpha_i, beta_i

    if it is the first iteration, pass in
        zero for beta_prev and v_prev
    """
    v_tmp = M_op(v_curr)
    # v_tmp = M_op @ v_curr
    alpha = v_curr @ v_tmp
    v_next = v_tmp - alpha * v_curr - beta_prev * v_prev
    beta = jnp.linalg.norm(v_next)
    v_next = v_next / beta
    return v_next, alpha, beta


def approx_min_eigvec(M, q, tol=1e-6):
    '''
        Runs Lanczos method to find (approx) minimum eigenvalue of
        linear operator/matrix defined by M
    '''
    np.random.seed(0)
    n = M.shape[0]
    T = min(q, n-1)
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    Q = np.zeros((n, T+1))
    alpha = np.zeros(T)
    beta = np.zeros(T)

    Q[:, 0] = v

    for i in range(T):
        Q[:, i+1] = M @ Q[:, i]
        alpha[i] = Q[:, i].T @ Q[:, i+1]
        if i == 0:
            Q[:, i+1] = Q[:, i+1] - alpha[i] * Q[:, i]
        else:
            Q[:, i+1] = Q[:, i+1] - alpha[i] * Q[:, i] - beta[i-1] * Q[:, i-1]

        beta[i] = np.linalg.norm(Q[:, i+1])

        if beta[i] < tol:
            break

        Q[:, i+1] = Q[:, i+1] / beta[i]

    # print(Q)
    # print(i)

    alpha_trunc = alpha[:i+1]
    beta_trunc = beta[:i]
    # test = np.diag(alpha_trunc) + np.diag(beta_trunc, k=1) + np.diag(beta_trunc, k=-1)
    # test = spa.diags([alpha_trunc, beta_trunc, beta_trunc], offsets=[0, 1, -1])
    # l_test, v_test = scipy_mineigval(test)
    # print('all eigs:', np.linalg.eigvals(test.todense()))

    # print('l_test:', l_test)
    # lambd, u = eigh_tridiagonal(alpha_trunc, beta_trunc, select='i', select_range=[0, 0])
    # lambd, all_u = eigh_tridiagonal(alpha_trunc, beta_trunc)
    # print(lambd, v_test.shape, u.shape)
    # print('alpha_trunc', alpha_trunc)
    # print('beta_trunc', beta_trunc)

    l_test, v_test = eigh_tridiagonal(alpha_trunc, beta_trunc, select='i', select_range=[0, 0])

    # diffs = v_test - all_u
    # print(diffs)
    # print('norms of diff:', np.linalg.norm(diffs[0]), np.linalg.norm(diffs[:, 0]))

    # xi = lambd[0]
    # u = all_u[:, 0]
    # u = all_u[:, 0].reshape(-1, 1)
    # print(v_test.shape, u.shape, (v_test-u).shape)
    # print('v_test - u norm:', np.linalg.norm(v_test - u))
    # v = Q[:, :i+1] @ u

    xi = l_test[0]

    v = Q[:, :i+1] @ v_test
    v = v / np.linalg.norm(v)

    # print(v_test - u)

    # print('v_test - u norm:', np.linalg.norm(v_test - u))
    # print('result v calcs:', v.T @ M @ v, 'v norm:', np.linalg.norm(v))

    return xi, v


def scipy_mineigval(M):
    return [np.real(x) for x in spa.linalg.eigs(M, which='SR', k=1)]


def main():
    np.random.seed(0)
    n = 100
    M = np.random.randn(n, n)
    M = (M + M.T) / 2

    # def mv(v):
    #     return M @ v
    # D = spa.linalg.LinearOperator((n, n), matvec=mv)

    spa_lambda, spa_v = scipy_mineigval(M)
    lanc_lambda, lanc_v = approx_min_eigvec(M, 21)
    # print(spa_v, lanc_v)
    print('scipy lambd:', spa_lambda[0])
    print('lanczos lambd:', lanc_lambda)
    print('eigval diff:', np.linalg.norm(spa_v - lanc_v))

    print(spa_v.T @ M @ spa_v)
    print(lanc_v.T @ M @ lanc_v)

    # print((M @ lanc_v).shape, (lanc_lambda * lanc_v).shape)
    diff_spa = (M @ spa_v) - (spa_lambda * spa_v)
    diff = (M @ lanc_v) - (lanc_lambda * lanc_v)
    print(np.linalg.norm(diff_spa), np.linalg.norm(diff))


if __name__ == '__main__':
    main()
