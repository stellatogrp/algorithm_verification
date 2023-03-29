import cvxpy as cp
# import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
from tqdm import trange
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
from algocert.solvers.sdp_cgal_solver.lanczos import lanczos
from jax.experimental import sparse


def cgal_solve_np():
    # cp_res = self.test_with_cvxpy()
    # print(cp_res)
    cp_res = 0
    # exit(0)
    # np.random.seed(0)
    alpha = 5
    beta_zero = 1
    A_op = self.estimate_A_op()
    K = np.inf
    n = self.problem_dim
    d = len(self.A_matrices)
    X = np.zeros((n, n))
    y = np.zeros(d)
    T = 500
    obj_vals = []
    X_resids = []
    y_resids = []
    feas_vals = []

    for t in trange(1, T+1):
        beta = beta_zero * np.sqrt(t + 1)
        eta = 2 / (t + 1)
        w = self.proj(self.AX(X) + y / beta)
        D = self.C_matrix + self.Astar_z(y + beta * (self.AX(X) - w))

        xi, v = self.minimum_eigvec(D)
        xi = np.real(xi[0])
        v = np.real(v)

        if xi < 0:
            H = alpha * np.outer(v, v)
        else:
            H = np.zeros(X.shape)

        # H = alpha * np.outer(v, v)

        Xnew = (1 - eta) * X + eta * H
        Xresid = np.linalg.norm(X - Xnew)
        X = Xnew

        beta_plus = beta_zero * np.sqrt(t + 2)

        # compute gamma
        wbar = self.proj(self.AX(X) + y / beta_plus)
        rhs = 4 * beta * (eta ** 2) * (alpha ** 2) * (A_op ** 2)
        gamma = rhs / np.linalg.norm(self.AX(X) - wbar) ** 2
        gamma = min(beta_zero, gamma)


        ynew = y + gamma * (self.AX(X) - wbar)
        yresid = np.linalg.norm(y-ynew)
        if np.linalg.norm(ynew) < K:
            y = ynew
        else:
            print('exceed')
        new_obj = np.trace(self.C_matrix @ X)
        obj_vals.append(new_obj)
        X_resids.append(Xresid)
        y_resids.append(yresid)
        feas_vals.append(self.proj_dist(self.AX(X)))


def cgal(A_op, C_op, A_star_op, b, alpha, T, m, n, lightweight=False):
    """
    jax implementation of the cgal algorithm to solve

    min Tr(CX)
        s.t. Tr(A_i X) = b_i i=1, ..., m
             X is psd

    Primitives:
        C_op(x) = C x
            R^n --> R^n
        A_op(u) = (Tr(A_1 X), ..., Tr(A_m X))
             = mathcal{A} vec(X)
                where mathcal{A} = [vec(A_1), ..., vec(A_m)]
            R^n --> R^m
        A_star_op(u, z) = A^*(z) u
            = sum_i z_i A_i u
            (R^n, R^m) --> R^n


    Algorithm: 3.1 of https://arxiv.org/pdf/1912.02949.pdf
    Init:
        beta_0 = 1
        K = inf
        X = zeros(n, n)
        y = zeros(m)
    for i in range(T):
        beta = beta_0 sqrt(i + 1)
        eta = 2 / (i + 1)
        q_t = t^{1/4} log n
        (lambda, v) = approxMinEvec(C + A^*(y + beta(AX -b)), q_t)
        X = (1 - eta) X + eta (alpha vv^T)
        y = y + gamma(AX - b)

    inputs:
        C_op: linear operator (see first primitive)
        A_op: linear operator (see second primitive)
        A_star_op: linear operator (see third primitive)
        b: right hand side vector (shape (m))
        T: number of iterations
        m: number of constraints
        n: number of rows of matrix of the standard form sdp
    outputs:
        X: primal solution - (n, n) matrix
        y: dual solution - (m) vector
    """

    # initialize cgal
    beta0, K, X_init, y_init = cgal_init(m, n)

    # cgal for loop
    final_val = cgal_for_loop(A_op, C_op, A_star_op, b, alpha, T,
                              X_init, y_init, beta0, jit=True, lightweight=lightweight)
    X, y, obj_vals, infeases, X_resids, y_resids = final_val
    return X, y, obj_vals, infeases, X_resids, y_resids


def cgal_init(m, n):
    beta0, K = 1, jnp.inf
    X, y = jnp.zeros((n, n)), jnp.zeros(m)

    return beta0, K, X, y


def cgal_for_loop(A_op, C_op, A_star_op, b, alpha, T, X_init, y_init, beta0,
                  jit=False, lightweight=False):
    m = b.size
    n = X_init.shape[0]
    partial_cgal_iter = partial(cgal_iteration,
                                C_op=C_op,
                                A_op=A_op,
                                A_star_op=A_star_op,
                                b=b,
                                alpha=alpha,
                                m=m,
                                n=n,
                                beta0=beta0,
                                lightweight=lightweight)
    obj_vals, infeases = jnp.zeros(T), jnp.zeros(T)
    X_resids, y_resids = jnp.zeros(T), jnp.zeros(T)
    init_val = X_init, y_init, obj_vals, infeases, X_resids, y_resids
    if jit:
        final_val = lax.fori_loop(0, T, partial_cgal_iter, init_val)
    else:
        final_val = python_fori_loop(0, T, partial_cgal_iter, init_val)
    X, y, obj_vals, infeases, X_resids, y_resids = final_val
    return X, y, obj_vals, infeases, X_resids, y_resids


def python_fori_loop(lower, upper, body_fun, init_val):
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def cgal_iteration(i, init_val, C_op, A_op, A_star_op, b, alpha, m, n, beta0, lightweight):
    X, y, obj_vals, infeases, X_resids, y_resids = init_val
    beta = beta0 * jnp.sqrt(i + 1)
    eta = 2 / (i + 1)
    # q = jnp.array(jnp.power(i, .25) * jnp.log(n), int)
    q = 20

    # get w
    # w = self.proj(self.AX(X) + y / beta)

    # create the new partial operator
    #   A_star_partial_op(u) = A_star(u, z)
    #   where z = y + beta(AX -b)
    z = y + beta * (A_op(X) - b)
    A_star_partial_op = partial(A_star_op, z=jnp.expand_dims(z, 1))

    # create new operator as input into lanczos
    def evec_op(u):
        # we take the negative since lobpcg_standard finds the largest evec
        return -C_op(u) - A_star_partial_op(u)

    # get minimum eigenvector
    # lambd, v = lanczos(evec_op, q, n)
    # import pdb
    # pdb.set_trace()
    lobpcg_out = sparse.linalg.lobpcg_standard(evec_op, jnp.ones((n, 1)))
    lambd, v = lobpcg_out[0], lobpcg_out[1]

    # if lambd < 0:
    #     H = 0 * X
    # else:
    #     H = alpha * jnp.outer(v, v)
    H = alpha * jnp.outer(v, v)

    # compute gamma
    gamma_rhs = 4 * (alpha ** 2) * beta * (eta ** 2)
    w = A_op(X) - b
    primal_infeas = jnp.linalg.norm(w)

    gamma = gamma_rhs / primal_infeas ** 2
    gamma = jnp.min(jnp.array([gamma, beta0]))

    # update primal and dual solutions with min evec
    X_next = (1 - eta) * X + eta * H
    y_next = y + gamma * w

    # compute progress and store it if lightweight is set to False
    if not lightweight:
        # obj_vals = obj_vals.at[i].set(C_op(X))
        # infeases = infeases.at[i].set(jnp.linalg.norm(A_op(X) - b))
        X_resids = X_resids.at[i].set(jnp.linalg.norm(X - X_next))
        y_resids = y_resids.at[i].set(jnp.linalg.norm(y - y_next))

    # update the val for the lax.fori_loop
    val = X_next, y_next, obj_vals, infeases, X_resids, y_resids
    return val


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
