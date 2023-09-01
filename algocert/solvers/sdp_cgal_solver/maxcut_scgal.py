import time

import cvxpy as cp

# import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as spa
from lanczos import approx_min_eigvec
from nymstrom import NymstromSketch
from tqdm import trange


class SCGALMaxCutTester(object):
    '''
        Solving min -tr(LX) s.t. diag(X) = 1, X >> 0
    '''

    def __init__(self, n, seed=0, scale=True):
        np.random.seed(seed)
        self.seed = seed
        self.n = n
        self.d = n
        self.m = int(.1 * n)
        self.scale = scale
        if self.scale:
            self.scale_X = n
            self.alpha = 1
        else:
            self.scale_X = 1
            self.alpha = n
        self.generate_laplacian()
        self.Aop_lb = 1
        self.b = np.ones(self.d) / self.scale_X

    def generate_laplacian(self):
        n, m, seed = self.n, self.m, self.seed
        # g = Graph.Barabasi(n, m, seed=seed)  # should be connected
        G = nx.barabasi_albert_graph(n, m, seed=seed)
        L = nx.laplacian_matrix(G)
        if self.scale:
            self.scale_C = spa.linalg.norm(L)
            L = L / self.scale_C
        self.L = L
        self.obj_C = -L

    def solve_maxcut_cvxpy(self):
        n, C = self.n, self.obj_C
        # alpha = self.alpha
        b = self.b
        print('----solving maxcut sdp with cvxpy----')
        X = cp.Variable((n, n), symmetric=True)
        obj = cp.Minimize(cp.trace(C @ X))
        constraints = [
            X >> 0,
            cp.diag(X) == b,
            # cp.trace(X) == alpha,
        ]
        prob = cp.Problem(obj, constraints)
        start = time.time()
        res = prob.solve(verbose=True, solver=cp.MOSEK)
        end = time.time()
        print('cvxpy time', end - start)
        print('cvxpy obj', res)
        # print('Xopt', np.round(X.value, 3))
        return res

    def C(self, x):
        return -self.L @ x

    def A(self, X):
        return np.diag(X)

    def Astar(self, z):
        return np.diag(z)

    def Astar_primitive(self, z, x):
        return np.multiply(z, x)

    def lanczos(self, M, q):
        return approx_min_eigvec(M, q)

    def scgal(self, R=20, T=1000):
        print('----solving maxcut with sketchy cgal----')
        n, d, alpha = self.n, self.d, self.alpha
        # C = -self.L
        b = self.b
        beta_0 = 1
        S = NymstromSketch(n, R)
        z = np.zeros(d)
        y = np.zeros(d)
        z_vals = [z]
        y_vals = [y]
        for t in trange(1, T+1):
            beta = beta_0 * np.sqrt(t + 1)
            eta = 2 / (t + 1)

            def mv(v):
                w = y + beta * (z - b)
                # Astar_w = self.Astar(w)
                # return self.C(v) + Astar_w @ v
                return self.C(v) + self.Astar_primitive(w, v)
            D = spa.linalg.LinearOperator((n, n), matvec=mv)
            lambd, v = self.lanczos(D, int(np.ceil((t ** .25) * np.log(n))))

            z = (1 - eta) * z + eta * self.A(alpha * (v @ v.T))
            gamma_rhs = 4 * (alpha ** 2) * beta * (eta ** 2)
            w = z - b
            primal_infeas = np.linalg.norm(w) * self.scale_X
            if self.scale:
                primal_infeas = primal_infeas / (1 + np.linalg.norm(b))
            gamma = gamma_rhs / primal_infeas ** 2
            gamma = min(gamma, beta_0)
            y = y + gamma * w

            S.rank_one_update(np.sqrt(alpha) * v, eta)

            z_vals.append(z)
            y_vals.append(y)
        U, Delta = S.reconstruct()
        X_test = U @ np.diag(Delta) @ U.T
        print('final obj:', np.trace(self.obj_C @ X_test))
        return 0, 0


def main():
    n = 100
    test = SCGALMaxCutTester(n, scale=True)
    # cp_obj = test.solve_maxcut_cvxpy()
    # print(test.scale_C)
    start = time.time()
    X_vals, y_vals = test.scgal(R=10, T=1000)
    end = time.time()
    print('time: ', end - start)
    # test.process_plot_resids(X_vals, y_vals, cp_obj)
    # test.eig_test()


if __name__ == '__main__':
    main()
