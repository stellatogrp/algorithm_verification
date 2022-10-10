import cvxpy as cp
import numpy as np
import scipy.sparse as spa


def solve_via_admm(C, A_eq_vals, b_eq_vals, A_ineq_vals, b_ineq_vals, psd_size=2):
    n = C.shape[0]
    # C_vec = C.flatten('F')
    # num_eq_cons = len(A_eq_vals)
    # num_slack_vars = len(A_ineq_vals)
    # Is = spa.eye(num_slack_vars)

    Aeq_blocks = []
    b_vals = []
    for i in range(len(A_eq_vals)):
        Aeq_i = A_eq_vals[i]
        beq_i = b_eq_vals[i]
        Aeq_i_vec = Aeq_i.flatten('F').reshape((1, -1))  # 'F' -> column major (Fortan-style)
        Aeq_blocks.append(Aeq_i_vec)
        b_vals.append(beq_i)
    Aeq = np.vstack(Aeq_blocks)
    # print(Aeq.shape)

    Aineq_blocks = []
    for j in range(len(A_ineq_vals)):
        Aineq_j = A_ineq_vals[j]
        bineq_j = b_ineq_vals[j]
        Aineq_j_vec = Aineq_j.flatten('F').reshape((1, -1))
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
        # print(Mk.shape)
        M_blocks.append(Mk)
    M = spa.vstack(M_blocks)
    # print(M.shape)

    Q = spa.vstack([A, -M])
    print(Q.shape)

    pad_amount = Q.shape[0] - len(b_vals)
    b_vals += [0] * pad_amount
    b_vals = np.array(b_vals)
    print(b_vals.shape)


def solve_via_cvxpy(C, A_eq_vals, b_eq_vals, A_ineq_vals, b_ineq_vals, psd_size=2):
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
    print(res)


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


def main():
    np.random.seed(0)
    n = 10
    eq_k = 50
    ineq_k = 10
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
        new_A = new_A_half
        A_eq_vals.append(new_A)
        b_eq_vals.append(np.trace(new_A @ X_test))
    for _ in range(ineq_k):
        new_A = np.random.randn(n, n)
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
