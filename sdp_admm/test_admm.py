import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
from tqdm import trange


def vec_to_mat(x, n):
    sqrt_2 = np.sqrt(2)
    try:
        assert x.shape[0] == n * (n + 1) / 2
    except AssertionError as e:
        print(x.shape, n * (n + 1) / 2)
        raise e
    curr = 0
    output_mat = np.zeros((n, n))
    for j in range(n):
        for i in range(0, j+1):
            val = x[curr]
            if i == j:
                output_mat[i, j] = val
            else:
                output_mat[i, j] = val / sqrt_2
                output_mat[j, i] = val / sqrt_2
            curr += 1
    return output_mat


def mat_to_vec(X, n):
    # extract the upper triangle
    sqrt_2 = np.sqrt(2)
    output_vec = []
    for j in range(n):
        for i in range(0, j+1):
            if i == j:
                output_vec.append(X[i, j])
            else:
                output_vec.append(X[i, j] * sqrt_2)
    return np.array(output_vec)


def solve_via_admm(C, A_eq_vals, b_eq_vals, A_ineq_vals, b_ineq_vals, psd_size=2):
    print('--------solving problem via admm--------')
    n = C.shape[0]
    # C_vec = C.flatten('F')
    C_vec = mat_to_vec(C, n)
    # num_eq_cons = len(A_eq_vals)
    # num_slack_vars = len(A_ineq_vals)
    # Is = spa.eye(num_slack_vars)

    Aeq_blocks = []
    b_vals = []
    for i in range(len(A_eq_vals)):
        Aeq_i = A_eq_vals[i]
        beq_i = b_eq_vals[i]
        # Aeq_i_vec = Aeq_i.flatten('F').reshape((1, -1))  # 'F' -> column major (Fortan-style)
        Aeq_i_vec = mat_to_vec(Aeq_i, n).reshape((1, -1))
        Aeq_blocks.append(Aeq_i_vec)
        b_vals.append(beq_i)
    Aeq = np.vstack(Aeq_blocks)
    # print(Aeq.shape)

    Aineq_blocks = []
    for j in range(len(A_ineq_vals)):
        Aineq_j = A_ineq_vals[j]
        bineq_j = b_ineq_vals[j]
        # Aineq_j_vec = Aineq_j.flatten('F').reshape((1, -1))
        Aineq_j_vec = mat_to_vec(Aineq_j, n).reshape((1, -1))
        Aineq_blocks.append(Aineq_j_vec)
        b_vals.append(bineq_j)
    Aineq = np.vstack(Aineq_blocks)
    # print(Aineq.shape)
    A = np.vstack([Aeq, Aineq])
    # print(A.shape)

    M_blocks = []
    In = np.eye(n)
    for k in range(n - psd_size + 1):
        Ek = build_Ej(In, range(k, k + psd_size))
        Mk = spa.kron(Ek, Ek)
        Mk_trunc = truncate_Mj(Mk, n, psd_size)
        # print(Mk.shape)
        # M_blocks.append(Mk)
        M_blocks.append(Mk_trunc)
    M = spa.vstack(M_blocks)
    print(M.shape)

    Q = spa.vstack([A, -M])
    print(Q.shape)

    pad_amount = Q.shape[0] - len(b_vals)
    b_vals += [0] * pad_amount
    b_vals = np.array(b_vals)
    print(b_vals.shape)

    def Pi_K(x):
        out = x.copy()
        eq_n = len(A_eq_vals)
        ineq_n = len(A_ineq_vals)
        out[:eq_n] = 0
        out[eq_n: eq_n + ineq_n] = np.maximum(out[eq_n: eq_n + ineq_n], 0)
        psd_start = eq_n + ineq_n
        # psd_sq = psd_size ** 2
        psd_sq = int(psd_size * (psd_size + 1) / 2)
        # print(psd_sq)
        for k in range(len(M_blocks)):
            # print('---')
            zk_start = psd_start + k * psd_sq
            zk_end = psd_start + (k + 1) * psd_sq
            zk = x[zk_start: zk_end]
            Zk_mat = vec_to_mat(zk, psd_size)
            projected_mat = psd_proj(Zk_mat)
            out[zk_start: zk_end] = mat_to_vec(projected_mat, psd_size)

        return out

    print(b_vals)
    admm_alg(C_vec, Q, b_vals, Pi_K, max_iter=3000)


def psd_proj(X):
    eigvals, eigvecs = np.linalg.eig(X)
    eigvals_clipped = np.maximum(eigvals, 0)
    return eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T


def admm_alg(c, Q, q, Pi_K, max_iter=1000):
    print('--starting admm--')
    alpha = 1
    eps = 1e-5
    rho = 1
    rho_inv = 1 / rho
    sigma = 1
    (m, n) = Q.shape
    first_iter = 0

    In = spa.eye(n)
    Im = spa.eye(m)
    lhs_mat = spa.bmat([
        [In, Q.T],
        [Q, -rho_inv * Im]
    ])

    # lhs_mat_indirect = sigma * In + rho * Q.T @ Q
    # lhs_mat_indirect = lhs_mat_indirect.todense()

    xk = np.zeros(n)
    wk = np.zeros(m)
    yk = np.zeros(m)
    lhs_mat = lhs_mat.todense()  # TODO: replace with sparse computations
    abs_dual_gaps = []
    primal_res = []
    dual_res = []
    for i in trange(max_iter):
        # print(i, 'obj:', c.T @ xk)
        # print(i, 'dual obj:', -q.T @ yk)
        # print('duality gap:', np.abs(c.T @ xk - q.T @ yk))
        # print('primal residual:', np.linalg.norm(Q @ xk + wk - q, np.inf))
        # print('dual residual:', np.linalg.norm(c - Q.T @ yk, np.inf))
        gap = np.abs(c.T @ xk - q.T @ yk)
        primal = np.linalg.norm(Q @ xk + wk - q, np.inf)
        dual = np.linalg.norm(c - Q.T @ yk, np.inf)

        abs_dual_gaps.append(gap)
        primal_res.append(primal)
        dual_res.append(dual)

        if first_iter == 0 and gap <= eps and primal <= eps and dual <= eps:
            first_iter = i

        rhs = np.concatenate([sigma * xk - c, q - wk + rho_inv * yk])
        kkt_sol = np.linalg.solve(lhs_mat, rhs)
        xtilde_kplus1 = kkt_sol[:n]
        nu_kplus1 = kkt_sol[n:]
        wtilde_kplus1 = wk - rho_inv * (nu_kplus1 + yk)
        x_kplus1 = alpha * xtilde_kplus1 + (1 - alpha) * xk
        w_kplus1 = Pi_K(alpha * wtilde_kplus1 + (1 - alpha) * wk + rho_inv * yk)
        y_kplus1 = yk + rho * (alpha * wtilde_kplus1 + (1 - alpha) * w_kplus1 - w_kplus1)

        xk = x_kplus1
        wk = w_kplus1
        yk = y_kplus1

        # rhs = sigma * xk - c + Q.T @ (rho * (q - wk) + yk)
        # x_kplus1 = np.linalg.solve(lhs_mat_indirect, rhs)
        # wtilde_kplus1 = q - Q @ x_kplus1
        # w_kplus1 = Pi_K(wtilde_kplus1 + rho_inv * yk)
        # y_kplus1 = yk + rho * (wtilde_kplus1 - w_kplus1)

        # xk = x_kplus1
        # wk = w_kplus1
        # yk = y_kplus1

    print('primal obj at end:', c.T @ xk)
    print('first iter val:', first_iter)
    skip = 20

    fig, ax = plt.subplots(figsize=(6, 4))
    N_vals = range(max_iter)[::skip]
    ax.plot(N_vals, abs_dual_gaps[::skip], label='abs duality gap', color='red')
    ax.plot(N_vals, primal_res[::skip], label='primal res', color='purple')
    ax.plot(N_vals, dual_res[::skip], label='dual_res', color='green')
    # ax.plot(N_vals, eps * np.ones(len(N_vals)), color='black', linestyle='dashed')
    ax.axhline(eps, color='black', linestyle='dashed')
    ax.axvline(first_iter, color='black')

    plt.title(f'ADMM progress, every {skip} iterations, $\\rho=${rho}, $\sigma=${sigma}, $\\alpha={alpha}$')
    plt.xlabel('$N$')
    plt.ylabel('$\ell_\infty$-norm')
    plt.yscale('log')
    plt.legend()
    plt.show()


def solve_via_cvxpy(C, A_eq_vals, b_eq_vals, A_ineq_vals, b_ineq_vals, psd_size=2):
    print('--------solving test via cvxpy--------')
    n = C.shape[0]
    X = cp.Variable((n, n), symmetric=True)
    constraints = []
    for i in range(len(A_eq_vals)):
        Ai = A_eq_vals[i]
        bi = b_eq_vals[i]
        constraints.append(cp.trace(Ai @ X) == bi)
    for i in range(len(A_ineq_vals)):
        Ai = A_ineq_vals[i]
        bi = b_ineq_vals[i]
        constraints.append(cp.trace(Ai @ X) <= bi)
    # constraints += [X >> 0]

    In = np.eye(n)
    for j in range(n - psd_size + 1):
        Ej = build_Ej(In, range(j, j + psd_size))
        # print(Ej)
        # Zk = X[i:i+2, i:i+2]
        Zj = Ej @ X @ Ej.T
        # print(Zj.shape)
        constraints += [Zj >> 0]
    prob = cp.Problem(cp.Minimize(cp.trace(C @ X)), constraints)
    res = prob.solve()
    print('cvxpy res:', res)


def build_Ej(In, indices):
    blocks = []
    # print(indices)
    for i in indices:
        # print(i)
        ei = In[i]
        blocks.append([ei])
    # print(blocks)
    Ej = np.vstack(blocks)
    return spa.csc_matrix(Ej)


def truncate_Mj(Mj, n, psd_size):
    # the original Mj matrix is meant for the full vec(X), but with only the upper triangle,
    # need to slice out the relevant columns
    # M_j has n^2 columns, and since we use the upper triangle of X, we partition into n equal groups
    # and take up to the 1st, 2nd, ..., nth columns as subblocks
    Mj = Mj.tocsc()
    blocks = []
    for j in range(n):
        # print(Mj[:, j])
        block = Mj[:, j * n: j * n + j + 1]
        # print('test shape:', test.shape)
        blocks.append(block)
    out = spa.hstack(blocks).tocsr()
    rows = []
    for i in range(psd_size):
        block = out[i * psd_size: i * psd_size + i + 1, :]
        rows.append(block)
        print(block.shape)
    out = spa.vstack(rows).tocsc()
    return out


def test_mat_vec():
    np.random.seed(0)
    n = 4
    X = np.random.randn(n, n)
    X = (X + X.T) / 2
    X_vec = mat_to_vec(X, n)
    # X_mat = vec_to_mat(X_vec, n)
    A = np.random.randn(n, n)
    A = (A + A.T) / 2
    A_vec = mat_to_vec(A, n)

    X_vec_flat = X.flatten('F')
    A_vec_flat = A.flatten('F')

    print('inner prod default:', A_vec_flat @ X_vec_flat)
    print('inner prod after vec:', A_vec @ X_vec)


def main():
    # test_mat_vec()
    np.random.seed(0)
    n = 10
    eq_k = 50
    ineq_k = 10
    # n = 3
    # eq_k = 5
    # ineq_k = 1
    C = np.random.randn(n, n)
    C = (C + C.T) / 2
    A_eq_vals = []
    b_eq_vals = []
    A_ineq_vals = []
    b_ineq_vals = []
    X_test_half = np.random.randn(n, n)
    X_test = X_test_half @ X_test_half.T
    for _ in range(eq_k):
        new_A_half = np.random.randn(n, n)
        # new_A = new_A_half @ new_A_half.T
        new_A = (new_A_half + new_A_half.T) / 2
        A_eq_vals.append(new_A)
        b_eq_vals.append(np.trace(new_A @ X_test))
    for _ in range(ineq_k):
        new_A_half = np.random.randn(n, n)
        new_A = (new_A_half + new_A_half.T) / 2
        A_ineq_vals.append(new_A)
        b_ineq_vals.append(np.trace(new_A @ X_test) + 1)
    # print(b_vals)
    # for i in range(n-1):
    #     print(X_test[i, i+1], X_test[i+1, i])
    # print(np.trace(C @ X_test))
    psd_size = 2
    solve_via_cvxpy(C, A_eq_vals, b_eq_vals, A_ineq_vals, b_ineq_vals, psd_size=psd_size)
    solve_via_admm(C, A_eq_vals, b_eq_vals, A_ineq_vals, b_ineq_vals, psd_size=psd_size)


if __name__ == '__main__':
    main()
