import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt
from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction
from scipy.stats import ortho_group


class UnconstrainedQuadraticProgram(object):

    def __init__(self, n=5, num_zero_eigs=0, mu=1, L=100, seed=0, centered=True):
        np.random.seed(seed)
        self.n = n
        self.num_zero_eigs = num_zero_eigs
        self.L = L
        self.mu = mu
        self.centered = centered
        self._generate_problem()

    def _generate_problem(self):
        n = self.n
        num_zero_eigs = self.num_zero_eigs
        L = self.L
        mu = self.mu
        centered = self.centered

        if centered:
            Q = np.eye(n)
        else:
            Q = ortho_group.rvs(n)
        P_eigs = np.zeros(n)
        if n-num_zero_eigs-2 > 0:
            mid_eigs = np.random.uniform(low=mu, high=L, size=n-num_zero_eigs-2)
            len_mid = mid_eigs.shape[0]
        else:
            mid_eigs = []
            len_mid = 0
        P_eigs[num_zero_eigs] = mu
        P_eigs[num_zero_eigs + 1: num_zero_eigs + 1 + len_mid] = mid_eigs
        P_eigs[-1] = L
        P = Q @ np.diag(P_eigs) @ Q.T
        self.Q = Q
        # print('eigvals of P:', np.round(np.linalg.eigvals(P), 4))

        self.P = (P + P.T) / 2

    def get_t_opt(self):
        return 2 / (self.mu + self.L)

    def f(self, x):
        return .5 * x.T @ self.P @ x

    def gradf(self, x):
        return self.P @ x


def solve_sdp(UQP, c, r, k=1):
    n = UQP.n
    P = UQP.P
    t = UQP.get_t_opt()
    x = cp.Variable(n)
    x_mat = cp.reshape(x, (n, 1))
    X = cp.Variable((n, n), symmetric=True)
    # C = np.
    ItP = np.eye(n) - t * P
    # print(np.linalg.matrix_power(ItP, 0))
    C = np.linalg.matrix_power(ItP, k) - np.linalg.matrix_power(ItP, k-1)
    CTC = C.T @ C
    # print(CTC)

    obj = cp.Maximize(cp.trace(CTC @ X))
    constraints = [
        cp.trace(X) <= r ** 2 + 2 * c.T @ x - c.T @ c,
        cp.bmat([
            [X, x_mat],
            [x_mat.T, np.array([[1]])]
        ]) >> 0,
    ]

    prob = cp.Problem(obj, constraints)
    res = prob.solve(solver=cp.SCS)
    # print(res)
    # print(X.value)
    U, sigma, VT = np.linalg.svd(X.value, full_matrices=False)
    # print('eigvals of X:', sigma)
    # print(U, U[:, 0])
    # print('Q:', UQP.Q)
    return res, U[:, 0] * np.sqrt(sigma[0])


def solve_centered_sdp(UQP, r, k=1):
    n = UQP.P.shape[0]
    return solve_sdp(UQP, np.zeros(n), r, k=k)


def UQP_pep(mu, L, r, t, k=1):
    verbose=0
    problem = PEP()
    func = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    xs = func.stationary_point()
    x0 = problem.set_initial_point()

    x = [x0 for _ in range(k + 1)]
    for i in range(k):
        x[i+1] = x[i] - t * func.gradient(x[i])

    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)

    problem.set_performance_metric((x[-1] - x[-2]) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    return pepit_tau


def sample_c(n, R):
    samp = np.random.randn(n)
    samp = R * samp / np.linalg.norm(samp)
    return samp


def off_centered_sdp(n, R, r, k, UQP):
    c = sample_c(n, R)
    print(np.round(c, 3))
    res, x = solve_sdp(UQP, c, r, k=k)
    # print(res)
    # print(c, x)
    if np.linalg.norm(c - x) >= r + 1e-3:
        # print('flipping')
        x = -x
    # print(np.linalg.norm(c), np.linalg.norm(c - x))
    return c, x


def gd(x0, t, k, UQP):
    out = [x0]
    curr = x0
    for _ in range(k):
        new = curr - t * UQP.gradf(curr)
        out.append(new)
        # print(new)
        curr = new
    return out


def experiment1():
    seed = 1
    n = 2
    mu = 1
    L = 10
    R = .9
    r = .1
    k = 1
    gd_k = 10

    # np.random.seed(seed)
    UQP = UnconstrainedQuadraticProgram(n, mu=mu, L=L, seed=seed, centered=True)
    res, x_wc = solve_centered_sdp(UQP, R + r, k=k)
    print(x_wc)
    c_wc = R * x_wc / np.linalg.norm(x_wc)
    # print('centered at 0 res:', res)
    c1, x1 = off_centered_sdp(n, R, r, k, UQP)
    c2, x2 = off_centered_sdp(n, R, r, k, UQP)
    # UQP_pep(mu, L, R+r, UQP.get_t_opt(), k=k)


    # PEP
    taus = []
    for k in range(1, gd_k + 1):
        tau = UQP_pep(mu, L, R + r, UQP.get_t_opt(), k=k)
        taus.append(tau)

    fp_resids = []
    for (x, c) in zip([x_wc, x1, x2], [c_wc, c1, c2]):

        gd_out = gd(x, UQP.get_t_opt(), gd_k, UQP)
        fp_resids.append([np.linalg.norm(gd_out[i+1] - gd_out[i]) ** 2 for i in range(gd_k)])

    wc_fp_resids = fp_resids[0]
    npt.assert_allclose(taus, wc_fp_resids, rtol=1e-4, atol=1e-4)
    npt.assert_array_less(fp_resids[1], taus)
    npt.assert_array_less(fp_resids[2], taus)


class TestBasicGD(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)

    def test_UQP_bounds(self):
        experiment1()

def main():
    experiment1()


if __name__ == '__main__':
    main()
