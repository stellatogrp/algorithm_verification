import cvxpy as cp
import numpy as np
from ISTA_class import ISTA

# from PEPit.examples.composite_convex_minimization.proximal_gradient import wc_proximal_gradient


def sample_l2ball(c, r, N):
    all_samples = []
    for _ in range(N):
        sample = np.random.normal(0, 1, c.shape[0])
        sample = np.random.uniform(0, r) * sample / np.linalg.norm(sample)
        # print(np.linalg.norm(sample))
        # print(sample.reshape(-1, 1) + c)
        all_samples.append(sample.reshape(-1, 1) + c)
    return all_samples


def b_to_xopt(instance, b_vals):
    out = []
    A = instance.A
    lambd = instance.lambd

    for b in b_vals:
        x = cp.Variable(A.shape[1])
        obj = cp.Minimize(.5 * cp.sum_squares(A @ x - b.reshape(-1, )) + lambd * cp.sum(cp.abs(x)))
        prob = cp.Problem(obj, [])
        prob.solve()
        out.append(x.value)

    return out


def S(v, t):
    pass


def ISTA_solve(instance, b):
    pass


def c_to_ISTA(instance, b_vals, K=7):
    pass


def sample_and_run(instance, b_c, b_r, N, K=7):
    b_vals = sample_l2ball(b_c, b_r, N)
    x_opt_vals = b_to_xopt(instance, b_vals)

    np.linalg.norm(max(x_opt_vals, key=lambda x: np.linalg.norm(x)))

    c_to_ISTA(instance, b_vals, K=K)


def main():
    m, n = 20, 15
    b_c = 10 * np.ones((m, 1))
    b_r = .5

    instance = ISTA(m, n, b_c, b_r, lambd=5, seed=1)
    instance.test_cp_prob()
    print(instance.get_t_opt())

    np.random.seed(0)
    # sample_and_run(instance, b_c, b_r, N)


if __name__ == '__main__':
    main()
