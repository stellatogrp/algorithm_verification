import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from algocert.solvers.admm_chordal import chordal_solve, unvec_symm
import cvxpy as cp


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
    assert iter_losses[-1] < 1e-4 and iter_losses[0] > 1
    assert jnp.abs(prob.value - obj) <= 1e-5


def test_chordal_sdp():
    pass
