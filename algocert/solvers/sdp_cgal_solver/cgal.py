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


def scale_problem_data(C_op_orig, A_op_orig, A_star_op_orig, alpha_orig, norm_A_orig, b_orig,
                       scale_x, scale_c, scale_a):
    """
    given original problem data: (C_op_orig, A_op_orig, A_star_op_orig, alpha_orig, b_orig)
    this method scales the problem data so that it is more favorable for cgal

    inputs: C_op_orig, A_op_orig, A_star_op_orig, alpha_orig, b_orig, scale_x, scale_c
    outputs: C_op, A_op, A_star_op, alpha, b, rescale_obj, rescale_feas
    """

    def C_op(x):
        return C_op_orig(x) * scale_c

    def A_op(u):
        return A_op_orig(u) * scale_a

    def A_star_op(u, z):
        return A_star_op_orig(scale_a * u, z)

    b = b_orig * scale_x
    alpha = alpha_orig * scale_x

    norm_A = norm_A_orig * scale_a

    rescale_obj = 1 / (scale_c * scale_x)
    rescale_feas = 1 / (scale_a * scale_x)

    scaled_data = dict(C_op=C_op, A_op=A_op, A_star_op=A_star_op, alpha=alpha, norm_A=norm_A,
                       b=b, rescale_obj=rescale_obj, rescale_feas=rescale_feas)
    return scaled_data


def recover_original_sol(X_scaled, y_scaled, scale_x, scale_c, scale_a):
    """
    given a solution of cgal of the scaled problem this returns a solution to the original problem
        by reverseing the scaling
    """
    X = X_scaled / scale_x
    y = scale_a * y_scaled / scale_c
    return X, y


def cgal(A_op, C_op, A_star_op, b, alpha, norm_A, rescale_obj, rescale_feas, cgal_iters, m, n, beta0=1, y_max=jnp.inf,
         lobpcg_iters=100, lobpcg_tol=1e-10, warm_start_v=True, jit=True, lightweight=False):
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
    X_init, y_init, z_init = cgal_init(m, n)

    # cgal for loop
    cgal_out = cgal_for_loop(A_op, C_op, A_star_op, b, alpha, norm_A, rescale_obj, rescale_feas,
                             cgal_iters,
                             X_init, y_init, z_init,
                             beta0, y_max,
                             jit=jit,
                             lobpcg_iters=lobpcg_iters,
                             lobpcg_tol=lobpcg_tol,
                             warm_start_v=warm_start_v, lightweight=lightweight)
    return cgal_out


def cgal_init(m, n):
    X, y, z = jnp.zeros((n, n)), jnp.zeros(m), jnp.zeros(m)
    return X, y, z


def cgal_for_loop(A_op, C_op, A_star_op, b, alpha, norm_A, rescale_obj, rescale_feas,
                  cgal_iters, X_init, y_init, z_init, beta0, y_max,
                  jit, lobpcg_iters, lobpcg_tol, warm_start_v, lightweight):
    m = b.size
    n = X_init.shape[0]
    def proj(input):
        return b
    static_dict = dict(C_op=C_op,
                       A_op=A_op,
                       A_star_op=A_star_op,
                       b=b,
                       alpha=alpha,
                       norm_A=norm_A,
                       proj=proj,
                       rescale_obj=rescale_obj,
                       rescale_feas=rescale_feas,
                       m=m,
                       n=n,
                       beta0=beta0,
                       y_max=y_max,
                       lobpcg_iters=lobpcg_iters,
                       lobpcg_tol=lobpcg_tol,
                       warm_start_v=warm_start_v,
                       lightweight=lightweight)
    partial_cgal_iter = partial(cgal_iteration, static_dict=static_dict)
    obj_vals, infeases = jnp.zeros(cgal_iters), jnp.zeros(cgal_iters)
    X_resids, y_resids = jnp.zeros(cgal_iters), jnp.zeros(cgal_iters)
    lobpcg_steps_mat = jnp.zeros(cgal_iters)
    v_init = jnp.ones((n, 1))
    init_val = X_init, y_init, z_init, obj_vals, infeases, X_resids, y_resids, lobpcg_steps_mat, v_init
    if jit:
        final_val = lax.fori_loop(0, cgal_iters, partial_cgal_iter, init_val)
    else:
        final_val = python_fori_loop(0, cgal_iters, partial_cgal_iter, init_val)
    X, y, z, obj_vals, infeases, X_resids, y_resids, lobpcg_steps, v_final = final_val
    cgal_out = dict(X=X, y=y, obj_vals=obj_vals, infeases=infeases, X_resids=X_resids,
                    y_resids=y_resids, lobpcg_steps=lobpcg_steps)
    return cgal_out


def python_fori_loop(lower, upper, body_fun, init_val):
    """
    this method is meant as a copy of the jax.lax.fori_loop version
        we don't jit this
    used as a comparison and to make sure the jit is helping
    """
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def cgal_iteration(i, init_val, static_dict):
    # unpack static_dict which is meant to not change for an entire problem
    #   i and init_val will change and be passed to/from jax.lax.fori_loop
    C_op, A_op, A_star_op = static_dict['C_op'], static_dict['A_op'], static_dict['A_star_op']
    b, alpha, norm_A = static_dict['b'], static_dict['alpha'], static_dict['norm_A']
    rescale_obj, rescale_feas = static_dict['rescale_obj'], static_dict['rescale_feas']
    m, n = static_dict['m'], static_dict['n']
    beta0, y_max = static_dict['beta0'], static_dict['y_max']
    lobpcg_iters, lobpcg_tol = static_dict['lobpcg_iters'], static_dict['lobpcg_tol']
    warm_start_v, lightweight = static_dict['warm_start_v'], static_dict['lightweight']
    proj = static_dict['proj']

    # unpack init_val
    X, y, z, obj_vals, infeases, X_resids, y_resids, lobpcg_steps_mat, prev_v = init_val
    beta = beta0 * jnp.sqrt(i + 1)
    eta = 2 / (i + 1)

    w = proj(z + y / beta)
    a_star_z_fixed = y + beta * (z - w)
    A_star_partial_op = partial(A_star_op, z=jnp.expand_dims(a_star_z_fixed, 1))

    # create new operator as input into lanczos
    def evec_op(u):
        # we take the negative since lobpcg_standard finds the largest evec
        return -C_op(u) - A_star_partial_op(u)

    # get minimum eigenvector
    if warm_start_v:
        lobpcg_out = sparse.linalg.lobpcg_standard(evec_op, prev_v, m=lobpcg_iters, tol=lobpcg_tol)
    else:
        lobpcg_out = sparse.linalg.lobpcg_standard(evec_op, jnp.zeros((n, 1)), m=lobpcg_iters, tol=lobpcg_tol)

    lambd, v, lobpcg_steps = lobpcg_out[0], lobpcg_out[1], lobpcg_out[2]

    # we flip the sign because lobpcg_standard looks for the largest
    #   eigenvalue, eigenvector pair
    lambd = -lambd

    # this will be printed if jit set to false
    print('z', z)
    print('lambd', lambd)
    print('lobpcg_steps', lobpcg_steps)

    # update z
    v_alpha = jnp.sqrt(alpha) * v * (lambd < 0)
    vvT = jnp.outer(v_alpha, v_alpha)
    new_z_dir = A_op(vvT)
    z_next = (1 - eta) * z + eta * new_z_dir

    # calculate primal direction
    H = vvT

    # update primal
    X_next = (1 - eta) * X + eta * H

    # compute gamma
    gamma_rhs = (alpha ** 2) * beta * norm_A * (eta ** 2)

    # dual update
    w = z_next - proj(z_next + y / beta)
    primal_infeas = jnp.linalg.norm(w)

    gamma_raw = gamma_rhs / (primal_infeas ** 2)
    gamma = jnp.min(jnp.array([gamma_raw, beta0]))

    # update dual solutions with min evec
    #   reject if the new ||y_t|| > K
    y_temp = y + gamma * w
    y_next = y + gamma * w * (jnp.linalg.norm(y_temp) <= y_max)

    # update computationally cheap progress
    infeases = infeases.at[i].set(primal_infeas * rescale_feas)
    lobpcg_steps_mat = lobpcg_steps_mat.at[i].set(lobpcg_steps)

    # compute progress and store it if lightweight is set to False
    if not lightweight:
        obj_vals = obj_vals.at[i].set(jnp.trace(C_op(X)) * rescale_obj)
        # infeases = infeases.at[i].set(jnp.linalg.norm(A_op(X) - b))
        X_resids = X_resids.at[i].set(jnp.linalg.norm(X - X_next))
        y_resids = y_resids.at[i].set(jnp.linalg.norm(y - y_next))
        

    # update the val for the lax.fori_loop
    val = X_next, y_next, z_next, obj_vals, infeases, X_resids, y_resids, lobpcg_steps_mat, v
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