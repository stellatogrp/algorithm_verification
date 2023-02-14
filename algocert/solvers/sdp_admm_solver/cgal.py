import cvxpy as cp
# import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
from tqdm import trange


class CGALTester(object):

    def __init__(self, n, d=None, seed=0):
        np.random.seed(seed)
        self.n = n
        if d is None:
            self.d = n
        else:
            self.d = d
        self.A_matrices = []
        self.bl_values = []
        self.bu_values = []
        self.generate_problem()
        self.generate_Aop_lowerbound()
        self.alpha = 10
        self.model_alpha = 5

    def generate_problem(self):
        n, d = self.n, self.d
        C = np.random.randn(n, n)
        self.C = (C + C.T) / 2
        X_temp = np.random.randn(n, n)
        X_temp = .1 / np.sqrt(n) * X_temp @ X_temp.T
        for _ in range(d):
            Ai = np.random.randn(n, n)
            self.A_matrices.append((Ai + Ai.T) / 2)
            # self.bl_values = np.random.randn(d)
            # self.bu_values = self.bl_values + np.random.randint(2, size=d)
            AiX = np.trace(Ai @ X_temp)
            self.bu_values.append(AiX)
            self.bl_values.append(AiX - np.random.randint(2))

    def generate_Aop_lowerbound(self):
        n = self.n
        X_temp = np.random.randn(n, n)
        X = X_temp @ X_temp.T
        X = X / np.linalg.norm(X)  # frobenius norm for matrices
        z = self.A(X)
        self.Aop_lb = np.linalg.norm(z)

    def solve_model_cvxpy(self):
        print('----solving model problem with cvxpy----')
        print('uses tr(AX) = b constraints instead of intervals')
        n, d = self.n, self.d
        X = cp.Variable((n, n), symmetric=True)
        obj = cp.trace(self.C @ X)
        constraints = [X >> 0, cp.trace(X) == self.model_alpha]
        for i in range(d):
            Ai = self.A_matrices[i]
            bi = self.bu_values[i]
            constraints += [
                cp.trace(Ai @ X) == bi,
            ]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        res = prob.solve()
        print('obj result:', res)
        print('Xopt trace:', np.trace(X.value))
        return res

    def cgal_model(self, T=1000):
        print('----solving model problem with cgal----')
        n, d, alpha = self.n, self.d, self.model_alpha
        C = self.C
        b = np.array(self.bu_values)
        Aop_lb = self.Aop_lb
        beta_0 = 1
        # K = np.inf
        X = np.zeros((n, n))
        y = np.zeros(d)
        X_vals = [X]
        y_vals = [y]
        for t in trange(1, T+1):
            beta = beta_0 * np.sqrt(t + 1)
            eta = 2 / (t + 1)
            D = C + self.Astar(y + beta * (self.A(X) - b))
            lambd, v = self.min_eigvec(D)
            X = (1 - eta) * X + eta * alpha * (v @ v.T)
            gamma_rhs = 4 * (alpha ** 2) * beta_0 * (Aop_lb) ** 2 / (t + 1) ** 1.5
            w = self.A(X) - b
            gamma = gamma_rhs / np.linalg.norm(w) ** 2
            gamma = min(gamma, beta_0)
            y = y + gamma * w
            X_vals.append(X)
            y_vals.append(y)
        # print(np.trace(C @ X))
        return X_vals, y_vals

    def plot_model_resids(self, X_vals, y_vals, cp_obj):
        T = len(X_vals) - 1
        b = self.bu_values
        X_resids = [np.linalg.norm(X_vals[i+1] - X_vals[i]) for i in range(T)]
        y_resids = [np.linalg.norm(y_vals[i+1] - y_vals[i]) for i in range(T)]
        obj_vals = [np.trace(self.C @ X) for X in X_vals]
        feas_dists = [np.linalg.norm(self.A(X)-b) for X in X_vals]
        self.plot_resids(X_resids, y_resids, obj_vals, feas_dists, T, cp_obj)

    def A(self, X):
        out = [np.trace(Ai @ X) for Ai in self.A_matrices]
        return np.array(out)

    def Cx(self, x):
        return self.C @ x

    def Astar(self, z):
        n, d = self.n, self.d
        out = np.zeros((n, n))
        for i in range(d):
            out += z[i] * self.A_matrices[i]
        return out

    def proj_K(self, x):
        bl = np.array(self.bl_values)
        bu = np.array(self.bu_values)
        return np.minimum(bu, np.maximum(x, bl))

    def proj_Kdist(self, x):
        proj_x = self.proj_K(x)
        return np.linalg.norm(x - proj_x)

    def min_eigvec(self, M):
        # in the actual algorithm use lanczos, placeholder for now
        return [np.real(x) for x in spa.linalg.eigs(M, which='SR', k=1)]

    def cgal_interval(self, T=1000):
        print('----solving with cgal----')
        n, d, alpha = self.n, self.d, self.alpha
        Aop_lb = self.Aop_lb
        beta_0 = 1
        # K = np.inf
        X = np.zeros((n, n))
        y = np.zeros(d)
        X_vals = [X]
        y_vals = [y]
        for t in trange(1, T+1):
            beta = beta_0 * np.sqrt(t + 1)
            eta = 2 / (t + 1)
            z = self.A(X)
            w = self.proj_K(z + y / beta)
            D = self.C + self.Astar(y + beta * (z - w))
            lambd, v = self.min_eigvec(D)
            if lambd < 0:
                new = alpha * v @ v.T
                X = (1 - eta) * X + eta * new
            else:  # new part = 0
                X = (1 - eta) * X
            beta_plus = beta_0 * np.sqrt(t + 2)
            z = self.A(X)
            w = self.proj_K(z + y / beta_plus)
            rhs_vec = z - w
            gamma = beta * (eta ** 2) * (alpha ** 2) * (Aop_lb) ** 2 / np.linalg.norm(rhs_vec) ** 2
            gamma = min(beta_0, gamma)
            # print('gamma', gamma)
            y = y + gamma * rhs_vec
            # print('obj', np.trace(self.C @ X))
            # print('feas dist', self.proj_Kdist(z))
            X_vals.append(X)
            y_vals.append(y)
        return X_vals, y_vals

    def solve_interval_cvxpy(self):
        print('----solving with cvxpy----')
        n, d = self.n, self.d
        X = cp.Variable((n, n), symmetric=True)
        obj = cp.trace(self.C @ X)
        constraints = [X >> 0, cp.trace(X) <= self.alpha]
        for i in range(d):
            Ai = self.A_matrices[i]
            bl = self.bl_values[i]
            bu = self.bu_values[i]
            constraints += [
                cp.trace(Ai @ X) >= bl,
                cp.trace(Ai @ X) <= bu,
                # cp.trace(Ai @ X) == bu,
            ]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        res = prob.solve()
        print('obj result:', res)
        print('Xopt trace:', np.trace(X.value))
        self.cp_obj = res
        return res

    def plot_interval_resids(self, X_vals, y_vals, cp_obj):
        T = len(X_vals) - 1
        X_resids = [np.linalg.norm(X_vals[i+1] - X_vals[i]) for i in range(T)]
        y_resids = [np.linalg.norm(y_vals[i+1] - y_vals[i]) for i in range(T)]
        obj_vals = [np.trace(self.C @ X) for X in X_vals]
        feas_dists = [self.proj_Kdist(self.A(X)) for X in X_vals]
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
        plt.yscale('symlog')
        plt.legend()
        plt.show()

    def feas_dist(self, X_vals):
        z_values = [self.A(X) for X in X_vals]
        z_last = np.array(z_values[-1])
        # print(self.bu_values - last)
        # print(last - self.bl_values)
        print(z_last - self.bu_values)


def test_and_plot_model_problem(test_prob):
    res = test_prob.solve_model_cvxpy()
    X_vals, y_vals = test_prob.cgal_model(T=5000)
    test_prob.plot_model_resids(X_vals, y_vals, res)


def test_and_plot_interval_problem(test_prob):
    res = test_prob.solve_interval_cvxpy()
    X_vals, y_vals = test_prob.cgal_interval(T=10000)
    test_prob.plot_interval_resids(X_vals, y_vals, res)


def main():
    n = 10
    d = 35
    test = CGALTester(n, d=d)

    # test_and_plot_model_problem(test)
    test_and_plot_interval_problem(test)


if __name__ == '__main__':
    main()
