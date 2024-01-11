import numpy as np
from ISTA_class import ISTA


def S(v, t):
    return np.maximum(v-t, 0) - np.maximum(-v-t, 0)


def ISTA_solve(instance, zk, b, K, t=.01):
    A = instance.A
    lambd = instance.lambd
    n = A.shape[1]
    I = np.eye(n)
    lhs = I - t * A.T @ A

    # zk = np.zeros(n)
    out_zres = []

    b = b.reshape(-1,)
    for _ in range(K):
        znew = S(lhs @ zk + t * A.T @ b, lambd * t)
        out_zres.append(np.linalg.norm(zk - znew) ** 2)
        zk = znew

        print('ista obj:', .5 * np.linalg.norm(A @ zk - b) ** 2 + lambd * np.linalg.norm(zk, 1))
    # print(out_zres)

    print('ISTA final z:', np.round(zk, 4))
    return out_zres


def FISTA_solve(instance, zk, b, K, t=.01):
    A = instance.A
    lambd = instance.lambd
    n = A.shape[1]
    I = np.eye(n)
    lhs = I - t * A.T @ A

    # zk = np.zeros(n)
    wk = zk.copy()

    out_zres = []
    b = b.reshape(-1,)

    beta_k = 1

    # K = 1000

    # print(zk)
    # print(wk)
    print('z0:', zk)
    print('w0:', wk)
    for i in range(K):
        znew = S(lhs @ wk + t * A.T @ b, lambd * t)
        beta_new = .5 * (1 + np.sqrt(1 + 4 * beta_k ** 2))
        wnew = znew + (beta_k - 1) / beta_new * (znew - zk)

        out_zres.append(np.linalg.norm(zk - znew) ** 2)

        # print(f'----K={i+1}----')
        # print('zk:', znew)
        # print('wk:', wnew)
        # print('wmul:', (beta_k - 1) / beta_new)

        zk = znew
        beta_k = beta_new
        wk = wnew
        print('fista obj:', .5 * np.linalg.norm(A @ zk - b) ** 2 + lambd * np.linalg.norm(zk, 1))
    # print(np.round(zk, 4))
    # print(out_zres)
    print('FISTA final z:', np.round(zk, 4))
    return out_zres


def main():
    m, n = 10, 15
    b_cmul = 10
    b_c = b_cmul * np.ones((m, 1))

    # b_c = np.array(list(range(m))).reshape((-1, 1))
    b_r = .5
    lambd = 10
    # t = .01
    test_K = 7

    instance = ISTA(m, n, b_c, b_r, lambd=lambd, seed=3)

    print(instance.A)
    print(instance.get_t_opt())
    t = .05

    btest = instance.sample_c().reshape(-1,)
    x, _, _, _ = np.linalg.lstsq(instance.A, btest, rcond=None)
    print(x)

    # t = .1 * instance.get_t_opt()
    print('sample opt sol:')
    instance.test_cp_prob()

    b = instance.sample_c()
    # z0 = np.zeros(n)
    z0 = x

    ISTA_res = ISTA_solve(instance, z0, b, K=test_K, t=t)
    FISTA_res = FISTA_solve(instance, z0, b, K=test_K, t=t)

    print('ISTA res:', ISTA_res)
    print('FISTA res:', FISTA_res)

    exit(0)

    K = 5
    instance.generate_FISTA_CP(K=K, t=t)
    # out_sdp = sdp_CP.solve(solver_type='SDP_CUSTOM')

    glob_ista_CP = instance.generate_CP(K=K, t=t)
    out_ista_glob = glob_ista_CP.solve(solver_type='GLOBAL', add_bounds=True)

    glob_fista_CP = instance.generate_FISTA_CP(K=K, t=t)
    out_fista_glob = glob_fista_CP.solve(solver_type='GLOBAL', add_bounds=True)

    print('ista:', out_ista_glob)
    print('fista:', out_fista_glob)

    exit(0)

    print('w muls:', instance.generate_betas(5))

    print(glob_fista_CP.solver.handler.get_param_var_map())
    var_list = glob_fista_CP.solver.handler.iterate_list
    print(var_list)
    iterate_val_map = glob_fista_CP.solver.handler.get_iterate_var_map()

    # z = var_list[3]
    y, u, v, z, w =  var_list
    print(z)
    print(iterate_val_map[z].X)
    print(w)
    print(iterate_val_map[w].X)


if __name__ == '__main__':
    main()
