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
    v_vals = [0, v]
    rho_vals = [0]
    omega_vals = []
    # print(v)
    for i in range(1, T+1):
        # print(v)
        Mv = M @ v
        # print('v norm', np.linalg.norm(Mv))
        omega = v @ Mv
        omega_vals.append(omega)
        # print('omega', omega)
        vnew = Mv - omega * v - rho_vals[i - 1] * v_vals[i - 1]
        rho = np.linalg.norm(vnew)
        # print('rho', rho)
        if rho < tol:
            break
        vnew = vnew / rho
        rho_vals.append(rho)
        v_vals.append(vnew)
        v = vnew
    # print(len(v_vals), len(rho_vals), len(omega_vals))
    # print(rho_vals)
    # print(omega_vals)
    omega = omega_vals[1: i]
    rho = rho_vals[1: i-1]
    lambd, u = eigh_tridiagonal(omega, rho)
    min_eigval = np.min(lambd)
    scalars = u[0]
    eigvec = 0
    for j in range(0, i-1):
        eigvec += scalars[j] * v_vals[j]
    # print(eigvec)
    return min_eigval, eigvec

    # np.random.seed(0)
    # n = M.shape[0]
    # T = min(q, n-1)
    # # v1 = np.random.randn(n)
    # v1 = v_vals[1]
    # rho_vals = np.zeros(T+1)
    # omega_vals = np.zeros(T+1)
    # v_vals = np.zeros((n, T+2))
    # # print(v_vals)
    # v_vals[:, 1] = v1
    # # print(v_vals)
    # # print(v_vals[:, 1])
    # for i in range(1, T+1):
    #     vi = v_vals[:, i]
    #     vi1 = M @ vi
    #     omega_i = vi @ vi1
    #     omega_vals[i] = omega_i
    #     vi1 -= omega_i * vi
    #     if i > 1:
    #         vi1 -= rho_vals[i - 1] * v_vals[:, i-1]
    #     rho = np.linalg.norm(vi1)
    #     rho_vals[i] = rho
    #     if rho < tol:
    #         break
    #     vi1 /= rho
    #     v_vals[:, i+1] = vi1
    # # print(v_vals)
    # print(rho_vals)
    # print(omega_vals)


def scipy_mineigval(M):
    return [np.real(x) for x in spa.linalg.eigs(M, which='SR', k=1)]


def main():
    np.random.seed(0)
    n = 100
    M = np.random.randn(n, n)
    M = (M + M.T) / 2
    spa_lambda, spa_v = scipy_mineigval(M)
    lanc_lambda, lanc_v = approx_min_eigvec(M, 50)
    print('scipy lambd:', spa_lambda[0])
    print('lanczos lambd:', lanc_lambda)
    print('eigval diff:', np.linalg.norm(spa_v - lanc_v))


if __name__ == '__main__':
    main()
