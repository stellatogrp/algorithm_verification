import numpy as np
from scipy.stats import ortho_group


class UnconstrainedQuadraticProgram(object):

    def __init__(self, n=5, num_zero_eigs=0, mu=1, L=100, seed=0):
        np.random.seed(seed)
        self.n = n
        self.num_zero_eigs = num_zero_eigs
        self.L = L
        self.mu = mu
        self._generate_problem()

    def _generate_problem(self):
        n = self.n
        num_zero_eigs = self.num_zero_eigs
        L = self.L
        mu = self.mu

        Q = ortho_group.rvs(n)
        P_eigs = np.zeros(n)
        mid_eigs = np.random.uniform(low=mu, high=L, size=n-num_zero_eigs-2)
        len_mid = mid_eigs.shape[0]
        P_eigs[num_zero_eigs] = L
        P_eigs[num_zero_eigs + 1: num_zero_eigs + 1 + len_mid] = mid_eigs
        P_eigs[-1] = mu
        P = Q @ np.diag(P_eigs) @ Q.T
        # print('eigvals of P:', np.round(np.linalg.eigvals(P), 4))

        self.P = (P + P.T) / 2
