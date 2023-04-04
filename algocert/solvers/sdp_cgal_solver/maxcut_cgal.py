import time

import cvxpy as cp
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse as spa
from lanczos import approx_min_eigvec
from tqdm import trange


class CGALMaxCutTester(object):
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

    def min_eigvec(self, M):
        # in the actual algorithm use lanczos, placeholder for now
        return [np.real(x) for x in spa.linalg.eigs(M, which='SR', k=1)]

    def lanczos(self, M, q):
        return approx_min_eigvec(M, q)

    def cgal(self, T=1000):
        print('----solving maxcut with cgal----')
        n, d, alpha = self.n, self.d, self.alpha
        # C = -self.L
        b = self.b
        beta_0 = 1
        X = np.zeros((n, n))
        y = np.zeros(d)
        X_vals = [X]
        y_vals = [y]
        xi_diffs = []
        v_norm_diffs = []
        for t in trange(1, T+1):
            beta = beta_0 * np.sqrt(t + 1)
            eta = 2 / (t + 1)

            def mv(v):
                w = y + beta * (self.A(X) - b)
                # Astar_w = self.Astar(w)
                # return self.C(v) + Astar_w @ v
                return self.C(v) + self.Astar_primitive(w, v)
            D = spa.linalg.LinearOperator((n, n), matvec=mv)
            # lambd, v = self.min_eigvec(D)
            # lambd, v = self.lanczos(D, 100)
            lambd, v = self.lanczos(D, int(np.ceil((t ** .25) * np.log(n))))
            sp_lambd, sp_v = self.min_eigvec(D)
            xi_diffs.append(np.abs(sp_lambd[0] - lambd))
            v_norm_diffs.append(np.linalg.norm(sp_v - v))
            X = (1 - eta) * X + eta * alpha * (v @ v.T)
            # gamma_rhs = 4 * (alpha ** 2) * beta_0 * (Aop ** 2) / (t + 1) ** 1.5
            gamma_rhs = 4 * (alpha ** 2) * beta * (eta ** 2)
            w = self.A(X) - b
            # print('infeas', np.linalg.norm(w))
            primal_infeas = np.linalg.norm(w) * self.scale_X
            if self.scale:
                primal_infeas = primal_infeas / (1 + np.linalg.norm(b))
            gamma = gamma_rhs / primal_infeas ** 2
            gamma = min(gamma, beta_0)
            # print(gamma)
            y = y + gamma * w
            X_vals.append(X)
            y_vals.append(y)
            # print(np.trace(self.obj_C @ X))
        # print(np.trace(self.obj_C @ X))
        # print(X)
        # print(self.A(X))
        print('final feas:', self.proj_dist(X_vals[-1]))
        print('final obj:', np.trace(self.obj_C @ X_vals[-1]))
        return X_vals, y_vals, xi_diffs, v_norm_diffs

    def proj_dist(self, X):
        return np.linalg.norm(self.A(X) - self.b)

    def process_plot_resids(self, X_vals, y_vals, cp_obj):
        T = len(X_vals) - 1
        # b = self.b
        X_resids = [np.linalg.norm(X_vals[i+1] - X_vals[i]) for i in range(T)]
        y_resids = [np.linalg.norm(y_vals[i+1] - y_vals[i]) for i in range(T)]
        obj_vals = [np.trace(self.obj_C @ X) for X in X_vals]
        feas_dists = [self.proj_dist(X) for X in X_vals]
        self.plot_resids(X_resids, y_resids, obj_vals, feas_dists, T, cp_obj)

    def plot_resids(self, X_resids, y_resids, obj_vals, feas_dists, T, cp_obj):
        fig, ax = plt.subplots(figsize=(6, 4))
        # ax.plot(range(1, T+1), obj_vals, label='obj')
        ax.plot(range(1, T+1), X_resids, label='X resids')
        ax.plot(range(1, T+1), y_resids, label='y resids')
        ax.plot(range(T+1), obj_vals, label='obj vals')
        ax.plot(range(T+1), feas_dists, label='feas dists')
        ax.axhline(y=cp_obj, linestyle='--', color='black', label='cvxpy obj')
        ax.axhline(y=0, color='black')

        plt.xlabel('$t$')
        # plt.ylabel('values')
        plt.yscale('symlog')
        plt.legend()
        plt.show()

    def plot_sp_vs_lanc(self, xi_diffs, v_norm_diffs):
        T = len(xi_diffs)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(range(T), xi_diffs, label='val diff', color='red')
        ax.plot(range(T), v_norm_diffs, label='vec diff', color='blue')
        ax.axhline(y=2)
        plt.xlabel('t')
        plt.yscale('log')
        plt.legend()
        plt.title('maxcut scipy vs lanc')
        # print(np.array(xi_diffs) - np.array(v_norm_diffs))
        print(xi_diffs)
        plt.show()

    def eig_test(self):
        n = 10
        A = np.random.randn(n, n)
        A = (A + A.T) / 2
        lambd, v = self.min_eigvec(A)
        np_lambd, np_v = np.linalg.eigh(A)
        print('spa', lambd)
        print('np:', np_lambd)


def main():
    n = 100
    test = CGALMaxCutTester(n, scale=True)
    # cp_obj = test.solve_maxcut_cvxpy()
    # print(test.scale_C)
    start = time.time()
    X_vals, y_vals, xi_diffs, v_norm_diffs = test.cgal(T=500)
    end = time.time()
    print('time: ', end - start)
    # test.process_plot_resids(X_vals, y_vals, cp_obj)
    test.plot_sp_vs_lanc(xi_diffs, v_norm_diffs)
    # test.eig_test()


if __name__ == '__main__':
    main()
