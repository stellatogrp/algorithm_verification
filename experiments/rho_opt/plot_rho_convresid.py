import matplotlib.pyplot as plt
import numpy as np
from test_rho_convresid import (OSQP_cert_prob, generate_problem,
                                generate_rho_opt)

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def plot(rho_vals, obj_vals, rho_star, K):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(rho_vals, obj_vals)
    ax.axvline(x=rho_star, color='black')
    ax.set_yscale('log')
    plt.xlabel('rho')
    plt.ylabel('max l2 convergence residual')
    plt.title(f'K = {K}')
    # plt.show()

    plt.savefig('images/basicobjvsrhoK1.pdf')


def main():
    np.random.seed(4)
    n = 5
    P, A, c = generate_problem(n)
    rho_opt = generate_rho_opt(P, A)
    K = 1
    # print(rho)
    # exit(0)
    rho_vals = [1, 5, 10, rho_opt, 25, 100]
    vals = []

    for rho in rho_vals:
        res_g = OSQP_cert_prob(P, A, c, rho, np.zeros((n, 1)), 1 * np.ones((n, 1)),
                               K=K, solver="GLOBAL", minimize=False)
        vals.append(res_g[0])

    for rho, res_g in zip(rho_vals, vals):
        print('rho:', rho)
        print('res_g:', res_g)

    plot(rho_vals, vals, rho_opt, K)


if __name__ == '__main__':
    main()
