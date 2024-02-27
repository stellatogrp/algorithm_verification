import numpy as np
import scipy.sparse as spa
from scipy.linalg import eigh_tridiagonal


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

    alpha_trunc = alpha[:i+1]
    beta_trunc = beta[:i]

    l_test, v_test = eigh_tridiagonal(alpha_trunc, beta_trunc, select='i', select_range=[0, 0], tol=tol)

    xi = l_test[0]
    v = Q[:, :i+1] @ v_test
    v = v / np.linalg.norm(v)

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
