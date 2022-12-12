# import cvxpy
import numpy as np

# import scipy.linalg as sla
# import scipy.sparse as spa


class NetworkUtilMax(object):

    def __init__(self, n=6, seed=1):
        np.random.seed(seed)

        self.n = n
        self.m = 2 * n

        self.R = np.random.binomial(n=1, p=0.2, m=(self.m, self.n))
        self.c_sample = np.random.uniform(1, 2, size=self.m)
        self.s_sample = np.random.uniform(.1, 1, size=self.n)
        self.w_sample = np.random.uniform(0, 1, size=self.n)
