import time
from functools import partial

import jax
# import jax.experimental.sparse as jspa
import jax.numpy as jnp
import numpy as np
import tqdm


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


def test_normal_func():
    n = 3
    print(n)
    X = np.array([
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6],
    ], dtype='float64')
    # print(mat(vec(X)))
    # print(np.triu_indices(n))
    num_iter = 100000
    start = time.time()
    for _ in tqdm.tqdm(range(num_iter)):
        y = vec(X)
        Y = mat(y)
        Y
    end = time.time()
    print(f'took {(end - start):.2f} s')


@jax.jit
def jax_vec(S):
    n = S.shape[0]
    S = jnp.copy(S)
    S *= jnp.sqrt(2)
    S = S.at[jnp.arange(n), jnp.arange(n)].divide(jnp.sqrt(2))
    return S[jnp.triu_indices(n)]


@partial(jax.jit, static_argnames=('n',))
def jax_mat(s, n):
    # n = (jnp.sqrt(8 * len(s) + 1) - 1) / 2
    # n = jnp.int16(n)
    # m = len(s)
    # m = jnp.sqrt(8 * m + 1)
    # T = jnp.zeros((m, m))
    # T = m
    S = jnp.zeros((n, n))
    S = S.at[jnp.triu_indices(n)].set(s / jnp.sqrt(2))
    S += S.T
    S = S.at[jnp.arange(n), jnp.arange(n)].divide(jnp.sqrt(2))
    return S


def test_jit_func():
    X = jnp.array([
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6],
    ])
    print(X)
    # print(jit_vec(X))

    print('---- with jit ----')
    start = time.time()
    # jit_vec = jax.jit(jax_vec)
    # x = jit_vec(X)
    # print(x)

    # n = X.shape[0]

    n = int((np.sqrt(8 * len(jax_vec(X)) + 1) - 1) / 2)
    print('check:', n)

    # jit_mat = jax.jit(jax_mat, static_argnums=(1,))
    # X_test = jit_mat(x, n)
    # print(X_test)
    # print(X)

    num_iter = 100000
    for _ in tqdm.tqdm(range(num_iter)):
        # y = jit_vec(X)
        # Y = jit_mat(y, n)

        y = jax_vec(X)
        Y = jax_mat(y, n)
    print(y, Y)
    end = time.time()
    print(f'took {(end - start):.2f} s')


# @partial(jax.jit, static_argnames=('m',))
def get_Ep_mat(m, blocks, X):
    n = X.shape[0]
    Ep = jnp.zeros((m, n))
    i = 0
    for b in blocks:
        # for j in jnp.arange(b[0], b[1]):
        #     Ep = Ep.at[i, j].set(1)
        #     i += 1
        Ep = Ep.at[i: i + b[1] - b[0], b[0]: b[1]].set(jnp.eye(b[1] - b[0]))
        i += (b[1] - b[0])
    return Ep


@jax.jit
def get_Hp_mat(Ep):
    return jnp.kron(Ep, Ep)


def Hp_full_to_vec(Hp):
    m_sq, n_sq = Hp.shape
    m = int(jnp.sqrt(m_sq))
    n = int(jnp.sqrt(n_sq))

    mc2 = int(m * (m + 1) / 2)
    nc2 = int(n * (n + 1) / 2)

    # middle = jnp.zeros((m_sq, nc2))
    # j = 0
    # for i in range(n):
    #     middle = middle.at[:, j: j + n - i].set(Hp[:, i * n: i * n + n - i])
    #     j += (n - i)

    # print('mid:\n', middle)
    # out = jnp.zeros((mc2, nc2))
    # j = 0
    # for i in range(m):
    #     out = out.at[j: j + m - i, :].set(middle[i * m: i * m + m - i, :])
    #     j += (m - i)

    # return out

    middle = jnp.zeros((mc2, n_sq))
    j = 0
    for i in range(m):
        middle = middle.at[j: j + m - i, :].set(Hp[i * m: i * m + m - i, :])
        j += (m - i)

    print('mid:\n', middle)
    out = jnp.zeros((mc2, nc2))
    j = 0
    for i in range(n):
        out = out.at[:, j: j + n - i].set(middle[:, i * n: i * n + n - i])
        j += (n - i)

    return out


def map_ij_to_tril(i, j, n):
    # print('j: ', j)
    return int(n * (n + 1) / 2 - (n - j) * (n - j + 1) / 2 + i - j)


def Hp_vec_direct(blocks, m, n):
    mc2 = int(m * (m + 1) / 2)
    nc2 = int(n * (n + 1) / 2)

    out = jnp.zeros((mc2, nc2))

    curr_row = 0
    for i in range(len(blocks)):
        bi = blocks[i]
        for t in range(bi[0], bi[1]):
            num_ones = bi[1] - t
            # print('t:', t)
            # print(num_ones)
            # print('from:', map_ij_to_tril(t, t, n), 'to:', map_ij_to_tril(t+num_ones, t, n))
            extract_idx = map_ij_to_tril(t, t, n)
            # out = out.at[curr_row: curr_row + num_ones, extract_idx: extract_idx + num_ones].set(jnp.eye(num_ones))
            out = out.at[jnp.arange(curr_row, curr_row + num_ones),
                         np.arange(extract_idx, extract_idx + num_ones)].set(jnp.ones(num_ones))
            curr_row += num_ones
            for j in range(i + 1, len(blocks)):
                bj = blocks[j]
                num_ones = bj[1] - bj[0]
                extract_idx = map_ij_to_tril(bj[0], t, n)
                out = out.at[jnp.arange(curr_row, curr_row + num_ones),
                             np.arange(extract_idx, extract_idx + num_ones)].set(jnp.ones(num_ones))
                curr_row += num_ones

    return out


def test_Hp_mats():
    X = jnp.array([
        [1, 2, 3],
        [2, 4, 5],
        [3, 5, 6],
    ])

    X = jnp.array([
        [1, 2, 3, 4, 5],
        [2, 6, 7, 8, 9],
        [3, 7, 10, 11, 12],
        [4, 8, 11, 13, 14],
        [5, 9, 12, 14, 15],
    ])

    # X = jnp.array([
    #     [1, 2, 3, 4],
    #     [2, 5, 6, 7],
    #     [3, 6, 8, 9],
    #     [4, 7, 9, 10],
    # ])

    # blocks = [(0, 1), (2, 3)]
    # blocks = [(0, 2), (2, 3)]
    # blocks = [(1, 4)]
    blocks = [(0, 1), (2, 3), (4, 5)]

    Ep_size = 3
    n = X.shape[0]
    Ep = get_Ep_mat(Ep_size, blocks, X)
    # Hp = get_Hp_mat(Ep)
    x = jax_vec(X)
    # print(get_Ep_mat(2, blocks, X))
    print(x)
    print('true block:\n', Ep @ X @ Ep.T)
    # print('full Hp:\n', Hp)
    # print('check vec:\n', jax_mat(Hp @ x, Ep_size))
    # print(Hp.T @ Hp)
    # Hp_vec = Hp_full_to_vec(Hp)
    Hp_vec = Hp_vec_direct(blocks, Ep_size, n)
    print('Hp for tri x:\n', Hp_vec)
    print('res from vec block:\n', Hp_vec @ x)
    # print(Hp_vec.T @ Hp_vec)

    out = jax_mat(Hp_vec @ x, Ep_size)
    print(jnp.round(out, 1))


def test_ij_tril():
    n = 4
    # for i in range(n):
    #     for j in range(i, n):
    #         print(j, i, map_ij_to_tril(j, i, n))
    indices = np.triu_indices(n)
    for (j, i) in zip(indices[0], indices[1]):
        print(i, j, map_ij_to_tril(i, j, n))


if __name__ == '__main__':
    # main()
    # test_normal_func()
    # test_jit_func()
    test_Hp_mats()
    # test_ij_tril()
