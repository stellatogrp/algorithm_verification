import time

import cvxpy as cp
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest

from algocert.solvers.admm_chordal import (chordal_solve, psd_completion,
                                           unvec_symm)

# @pytest.mark.skip(reason="temp")


def test_full_sdp():
    n_orig = 10
    chordal_list_of_lists = [jnp.arange(n_orig)]
    A_mat_list = []
    for i in range(n_orig):
        Ai = jnp.zeros((n_orig, n_orig))
        Ai = Ai.at[i, i].set(1)
        A_mat_list.append(Ai)

    # random C
    C = np.random.normal(size=(n_orig, n_orig))
    C = (C + C.T) / 2
    l, u = jnp.ones(n_orig), jnp.ones(n_orig)
    # rho_vec = jnp.ones(n_orig)
    sol = chordal_solve(A_mat_list, C, l, u, chordal_list_of_lists, rho=1)
    z_k, iter_losses, z_all = sol
    nc2 = int(n_orig * (n_orig + 1) / 2)
    X_sol = unvec_symm(z_k[:nc2], n_orig)
    evals, evecs = jnp.linalg.eigh(X_sol)
    obj = jnp.trace(C @ X_sol)

    # plt.plot(iter_losses)
    # plt.yscale('log')
    # plt.show()

    # solve with cvxpy
    X = cp.Variable((n_orig, n_orig))
    constraints = [X >> 0, cp.diag(X) == 1]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve()
    assert iter_losses[-1] < 1e-3 and iter_losses[0] > 1
    assert jnp.abs((prob.value - obj) / jnp.abs(prob.value)) <= 1e-4
    assert evals.min() >= -1e-5


def test_block_arrow_general():
    diag_block_sizes = 10
    arrow_width = 10
    num_blocks = 4
    n_orig = num_blocks * diag_block_sizes + arrow_width
    A_dense_mat_list = []
    m = 5 * n_orig
    for i in range(m):
        # Ai = jnp.zeros((n_orig, n_orig))
        # Ai = Ai.at[i, i].set(1)
        Ai = jnp.array(np.random.normal(size=(n_orig, n_orig)))
        A_dense_mat_list.append(Ai + Ai.T)

    l, u = -jnp.ones(m), jnp.ones(m)

    # random C
    C_dense = np.random.normal(size=(n_orig, n_orig))
    # C_dense = (C_dense + C_dense.T) / 2
    C_dense = C_dense @ C_dense.T / 10
    # import pdb
    # pdb.set_trace()

    # create block_arrow
    block_arrow_mask, chordal_list_of_lists = create_block_arrow_mask(diag_block_sizes,
                                                                      arrow_width, num_blocks)

    # mask C, A_i's
    C = jnp.multiply(C_dense, block_arrow_mask)
    A_mat_list = []
    for i in range(m):
        Ai = A_dense_mat_list[i]
        A_mat_list.append(jnp.multiply(Ai, block_arrow_mask))

    # solve with chordal sparsity
    t0 = time.time()
    sol = chordal_solve(A_mat_list, C, l, u, chordal_list_of_lists, sigma=1e-6, rho=1, k=10000)
    chordal_time = time.time() - t0
    print(chordal_time)
    z_k, iter_losses, z_all = sol
    nc2 = int(n_orig * (n_orig + 1) / 2)
    X_sol = unvec_symm(z_k[:nc2], n_orig)
    evals, evecs = jnp.linalg.eigh(X_sol)
    obj = jnp.trace(C @ X_sol)

    # solve with cvxpy
    X = cp.Variable((n_orig, n_orig), symmetric=True)
    constraints = [X >> 0]
    for i in range(m):
        constraints.append(cp.trace(A_mat_list[i] @ X) <= u[i])
        constraints.append(cp.trace(A_mat_list[i] @ X) >= l[i])
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve(verbose=True)

    assert iter_losses[-1] <= 1e-4
    assert iter_losses[0] > 1
    assert jnp.abs((obj - prob.value) / jnp.abs(prob.value)) <= 1e-3
    for i in range(m):
        assert jnp.trace(A_mat_list[i] @ X_sol) <= u[i] + 1e-3
        assert jnp.trace(A_mat_list[i] @ X_sol) >= l[i] - 1e-3
    # import pdb
    # pdb.set_trace()


@pytest.mark.skip(reason="temp")
def test_block_arrow_diag_constraints():
    diag_block_sizes = 10
    arrow_width = 10
    num_blocks = 9
    n_orig = num_blocks * diag_block_sizes + arrow_width
    A_mat_list = []
    for i in range(n_orig):
        Ai = jnp.zeros((n_orig, n_orig))
        Ai = Ai.at[i, i].set(1)
        A_mat_list.append(Ai)

    l, u = jnp.ones(n_orig), jnp.ones(n_orig)

    # random C
    C_dense = np.random.normal(size=(n_orig, n_orig))
    C_dense = (C_dense + C_dense.T) / 2

    # create block_arrow
    block_arrow_mask, chordal_list_of_lists = create_block_arrow_mask(diag_block_sizes,
                                                                      arrow_width, num_blocks)

    # mask C
    C = jnp.multiply(C_dense, block_arrow_mask)

    # solve with chordal sparsity
    t0 = time.time()
    sol = chordal_solve(A_mat_list, C, l, u, chordal_list_of_lists, rho=1)
    chordal_time = time.time() - t0
    z_k, iter_losses, z_all = sol
    nc2 = int(n_orig * (n_orig + 1) / 2)
    X_sol = unvec_symm(z_k[:nc2], n_orig)
    evals, evecs = jnp.linalg.eigh(X_sol)
    obj = jnp.trace(C @ X_sol)

    # do psd completion
    X_psd = psd_completion(X_sol, chordal_list_of_lists)
    evals, evecs = jnp.linalg.eigh(X_psd)

    final_obj = jnp.trace(C @ X_psd)

    # solve with cvxpy
    X = cp.Variable((n_orig, n_orig), symmetric=True)
    constraints = [X >> 0, cp.diag(X) == 1]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve(verbose=True)

    plt.plot(iter_losses)
    plt.yscale('log')
    plt.show()

    assert jnp.abs(final_obj - obj) <= 1e-4
    assert jnp.abs((final_obj - prob.value) / jnp.abs(prob.value)) <= 1e-4
    assert iter_losses[-1] < 1e-4 and iter_losses[0] > 1
    assert jnp.linalg.norm(jnp.diag(X_psd) - 1) <= 1e-4
    assert evals.min() >= -1e-2

    # solve without chordal sparsity
    t0 = time.time()
    sol_no_chordal = chordal_solve(A_mat_list, C, l, u, [jnp.arange(n_orig)], rho=1)
    non_chordal_time = time.time() - t0
    z_k_no_chordal, iter_losses_no_chordal, z_all_no_chordal = sol_no_chordal
    X_sol_no_chordal = unvec_symm(z_k_no_chordal[:nc2], n_orig)
    evals_no_chordal, evecs_no_chordal = jnp.linalg.eigh(X_sol_no_chordal)
    obj_no_chordal = jnp.trace(C @ X_sol_no_chordal)

    assert jnp.abs((obj_no_chordal - obj) / jnp.abs(obj_no_chordal)) <= 1e-3
    assert evals_no_chordal.min() >= -1e-3
    assert non_chordal_time >= 1.1 * chordal_time


def create_block_arrow_mask(diag_block_sizes, arrow_width, num_blocks):
    total_size = num_blocks * diag_block_sizes + arrow_width
    block_arrow_mask = jnp.zeros((total_size, total_size))
    chordal_list_of_lists = []
    arrow_indices = jnp.arange(num_blocks * diag_block_sizes, total_size)

    # make diagonal blocks take values of 1
    for i in range(num_blocks):
        start_index = i * diag_block_sizes
        end_index = (i + 1) * diag_block_sizes
        block_arrow_mask = block_arrow_mask.at[start_index:end_index,
                                               start_index:end_index].set(1)
        curr_chord = jnp.arange(start_index, end_index)
        chordal_list_of_lists.append(jnp.concatenate([curr_chord, arrow_indices]))

    # make arrow block take values of 1
    block_arrow_mask = block_arrow_mask.at[-arrow_width:, :].set(1)
    block_arrow_mask = block_arrow_mask.at[:, -arrow_width:].set(1)
    return block_arrow_mask, chordal_list_of_lists
