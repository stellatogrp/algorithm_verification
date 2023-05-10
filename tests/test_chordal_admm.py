import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from algocert.solvers.admm_chordal import chordal_solve, unvec_symm, psd_completion
import cvxpy as cp
import pytest

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
    rho_vec = jnp.ones(n_orig)
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
    assert jnp.abs((prob.value - obj) / jnp.abs(prob.value)) <= 1e-5
    assert evals.min() >= -1e-5


def test_block_arrow():
    diag_block_sizes = 10
    arrow_width = 10
    num_blocks = 4
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
    sol = chordal_solve(A_mat_list, C, l, u, chordal_list_of_lists, rho=1)
    z_k, iter_losses, z_all = sol
    nc2 = int(n_orig * (n_orig + 1) / 2)
    X_sol = unvec_symm(z_k[:nc2], n_orig)
    evals, evecs = jnp.linalg.eigh(X_sol)
    obj = jnp.trace(C @ X_sol)

    # do psd completion
    X_psd = psd_completion(X_sol, chordal_list_of_lists)
    final_obj = jnp.trace(C @ X_psd)

    # solve without chordal sparsity
    X = cp.Variable((n_orig, n_orig), symmetric=True)
    constraints = [X >> 0, cp.diag(X) == 1]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    prob.solve()

    assert jnp.abs(final_obj - obj) <= 1e-4
    assert jnp.abs((final_obj - prob.value) / jnp.abs(prob.value)) <= 1e-4
    assert iter_losses[-1] < 1e-4 and iter_losses[0] > 1
    assert jnp.linalg.norm(jnp.diag(X_psd) - 1) <= 1e-4
    



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
