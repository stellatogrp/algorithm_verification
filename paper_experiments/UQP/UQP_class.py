import numpy as np
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
        P_eigs[num_zero_eigs] = L
        P_eigs[num_zero_eigs + 1: num_zero_eigs + 1 + len_mid] = mid_eigs
        P_eigs[-1] = mu
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
