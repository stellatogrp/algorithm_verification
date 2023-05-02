import time

import cvxpy as cp
# import jax.experimental.sparse as jspa
# import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
from tqdm import trange

from algocert.solvers.sdp_cgal_solver import (OBJ_CANON_METHODS,
                                              SET_CANON_METHODS,
                                              STEP_CANON_METHODS)
from algocert.solvers.sdp_cgal_solver.lanczos import approx_min_eigvec, lanczos
from algocert.solvers.sdp_cgal_solver.nymstrom import NymstromSketch


class SDPCGALHandler(object):

    def __init__(self, CP, **kwargs):
        self.CP = CP
        self.K = CP.K  # number of steps to certify, TODO: change CP.N to CP.K
        self.num_samples = CP.num_samples
        self.alg_steps = CP.get_algorithm_steps()
        self.iterate_list = []
        self.param_list = []
        self.iterate_to_id_map = {}
        self.id_to_iterate_map = {}
        self.init_iter_range_map = {}
        self.sample_iter_bound_map = {}
        self.range_marker = 0
        self.problem_dim = 0
        self.A_matrices = []
        self.A_norms = []
        self.b_lowerbounds = []
        self.b_upperbounds = []
        self.C_matrix = None
        if 'scale' in kwargs:
            self.scale = kwargs['scale']
        else:
            self.scale = False

    def convert_hl_to_basic_steps(self):
        pass

    def create_iterate_id_maps(self):
        steps = self.CP.get_algorithm_steps()
        for i, step in enumerate(steps):
            iterate = step.get_output_var()
            self.iterate_to_id_map[iterate] = i
            self.id_to_iterate_map[i] = iterate
            self.iterate_list.append(iterate)

    def create_init_iterate_range_maps(self):
        for init_set in self.CP.get_init_sets():
            init_iter = init_set.get_iterate()
            dim = init_iter.get_dim()
            bounds = (self.range_marker, self.range_marker + dim)
            self.range_marker += dim
            self.init_iter_range_map[init_iter] = bounds

    def create_sample_iter_bound_maps(self):
        steps = self.CP.get_algorithm_steps()
        for i in range(self.num_samples):
            sample_dict = {}
            init_dict = {}
            for init_set in self.CP.get_init_sets():
                init_iter = init_set.get_iterate()
                init_dict[init_iter] = self.init_iter_range_map[init_iter]
            sample_dict[0] = init_dict

            for k in range(1, self.K + 1):
                step_dict = {}
                for j, step in enumerate(steps):
                    output_var = step.get_output_var()
                    dim = output_var.get_dim()
                    bounds = (self.range_marker, self.range_marker + dim)
                    self.range_marker += dim
                    step_dict[output_var] = bounds
                sample_dict[k] = step_dict

            for param_set in self.CP.get_parameter_sets():
                param_var = param_set.get_iterate()
                dim = param_var.get_dim()
                bounds = (self.range_marker, self.range_marker + dim)
                self.range_marker += dim
                sample_dict[param_var] = bounds

            # print(sample_dict)
            self.sample_iter_bound_map[i] = sample_dict

        print(self.sample_iter_bound_map)

    def set_problem_dim(self):
        self.problem_dim = self.range_marker + 1

    def create_lower_right_constraint(self):
        A = spa.lil_matrix((self.problem_dim, self.problem_dim))
        A[-1, -1] = 1
        self.A_matrices.append(A.tocsr())
        self.b_lowerbounds.append(1)
        self.b_upperbounds.append(1)

    def canonicalize_initial_sets(self):
        for init_set in self.CP.get_init_sets():
            canon_method = SET_CANON_METHODS[type(init_set)]
            A, b_l, b_u = canon_method(init_set, self)
            self.A_matrices += A
            self.b_lowerbounds += b_l
            self.b_upperbounds += b_u

    def canonicalize_parameter_sets(self):
        for param_set in self.CP.get_parameter_sets():
            canon_method = SET_CANON_METHODS[type(param_set)]
            A, b_l, b_u = canon_method(param_set, self)
            self.A_matrices += A
            self.b_lowerbounds += b_l
            self.b_upperbounds += b_u
        # print(len(self.A_matrices), len(self.b_lowerbounds), len(self.b_upperbounds))
        # print(self.b_lowerbounds, self.b_upperbounds)
        # print(self.A_matrices)

    def canonicalize_steps(self):
        steps = self.CP.get_algorithm_steps()
        for k in range(1, self.K + 1):
            for step in steps:
                canon_method = STEP_CANON_METHODS[type(step)]
                A, b_l, b_u = canon_method(step, k, self)
                self.A_matrices += A
                self.b_lowerbounds += b_l
                self.b_upperbounds += b_u

    def canonicalize_objective(self):
        self.C_matrix = spa.lil_matrix((self.problem_dim, self.problem_dim))
        if type(self.CP.objective) != list:
            obj_list = [self.CP.objective]
        else:
            obj_list = self.CP.objective

        for obj in obj_list:
            obj_canon = OBJ_CANON_METHODS[type(obj)]
            C_temp = obj_canon(obj, self)
            # print(C_temp)
            self.C_matrix += C_temp

        # Flipping objective to make the problem a minimization
        self.C_matrix = -self.C_matrix.tocsr()
        # print(self.C_matrix)
        if self.scale:
            self.C_scale = spa.linalg.norm(self.C_matrix)
            self.C_matrix /= self.C_scale
        else:
            self.C_scale = 1

    def divide_b_arrays_scalar(self, scalar):
        # for i in range(len(self.b_lowerbounds)):
        #     self.b_lowerbounds[i] /= scalar
        self.b_lowerbounds /= scalar
        self.b_upperbounds /= scalar

    def canonicalize(self):
        self.convert_hl_to_basic_steps()
        self.create_iterate_id_maps()
        self.create_init_iterate_range_maps()
        self.create_sample_iter_bound_maps()

        # any other preprocessing here

        self.set_problem_dim()
        self.create_lower_right_constraint()

        self.canonicalize_initial_sets()
        self.canonicalize_parameter_sets()
        self.canonicalize_steps()

        self.b_lowerbounds = np.array(self.b_lowerbounds)
        self.b_upperbounds = np.array(self.b_upperbounds)

        if self.scale:
            for i in range(len(self.A_matrices)):
                Ai = self.A_matrices[i]
                bl_i = self.b_lowerbounds[i]
                bu_i = self.b_upperbounds[i]
                # print(spa.linalg.norm(Ai))
                Ai_norm = spa.linalg.norm(Ai)
                self.A_norms.append(Ai_norm)
                self.A_matrices[i] = Ai / Ai_norm
                self.b_lowerbounds[i] = bl_i / Ai_norm
                self.b_upperbounds[i] = bu_i / Ai_norm
            # just check that it works:
            # for i in range(len(self.A_matrices)):
            #     Ai = self.A_matrices[i]
            #     print(spa.linalg.norm(Ai))
        self.canonicalize_objective()

    def test_with_cvxpy(self):
        X = cp.Variable((self.problem_dim, self.problem_dim), symmetric=True)
        # print(len(self.A_matrices), len(self.b_lowerbounds), len(self.b_upperbounds))
        obj = cp.trace(self.C_matrix @ X)
        constraints = [X >> 0]
        for i in range(len(self.A_matrices)):
            Ai = self.A_matrices[i]
            b_li = self.b_lowerbounds[i]
            b_ui = self.b_upperbounds[i]
            if b_li == -np.inf:
                b_li = -1000
            if b_ui == np.inf:
                b_ui = 1000
            constraints += [
                # cp.trace(X) <= 5,
                cp.trace(Ai @ X) >= b_li,
                cp.trace(Ai @ X) <= b_ui,
            ]
        prob = cp.Problem(cp.Minimize(obj), constraints)
        res = prob.solve(solver=cp.MOSEK)
        # print('res:', res)
        print('trace of X:', cp.trace(X).value)
        return res

    def estimate_A_op(self, num_tries=1000):
        '''
        https://math.stackexchange.com/questions/2747959/frobenius-and-operator-norms-of-rank-1-matrices
        '''
        op_norm = 0
        for _ in range(num_tries):
            # X = np.random.rand(self.problem_dim, self.problem_dim)
            # X = X @ X.T
            # X = X / np.linalg.norm(X)
            x = np.random.rand(self.problem_dim)
            x /= np.linalg.norm(x)
            X = np.outer(x, x)
            z = self.AX(X)
            test = np.linalg.norm(z)
            if test > op_norm:
                op_norm = test
        print('estimated A_op:', op_norm)
        return op_norm

    def AX(self, X):
        out = []
        for A in self.A_matrices:
            out.append(np.trace(A @ X))
        return np.array(out)

    def Astar_z(self, z):
        # print(z)
        outmat = spa.lil_matrix((self.problem_dim, self.problem_dim))
        for j, A in enumerate(self.A_matrices):
            outmat += z[j] * A
        return outmat

    def proj(self, x):
        bl = np.array(self.b_lowerbounds)
        bu = np.array(self.b_upperbounds)
        return np.minimum(bu, np.maximum(x, bl))

    def proj_dist(self, x):
        # bl = np.array(self.b_lowerbounds)
        # bu = np.array(self.b_upperbounds)
        # dist = 0
        # for i in range(len(x)):
        #     li = bl[i]
        #     ui = bu[i]
        #     xi = x[i]
        #     if li > xi:
        #         dist += li - xi
        #     elif ui < xi:
        #         dist += xi - ui
        # return dist
        proj_x = self.proj(x)
        return np.linalg.norm(x - proj_x)

    def minimum_eigvec(self, X):
        return spa.linalg.eigs(X, which='SR', k=1)

    def lanczos(self, M, q):
        return approx_min_eigvec(M, q)
        # return lanczos(M, q, M.shape[0])

    def lanczos_jax(self, M, q):
        return lanczos(M, q, self.problem_dim)

    def warmstartX_vec(self):
        out = np.zeros((self.problem_dim, 1))
        out[-1, 0] = 1
        # first, initsets:
        for init_set in self.CP.get_init_sets():
            x = init_set.get_iterate()
            sample = init_set.sample_point()
            (x_l, x_u) = self.init_iter_range_map[x]
            # print(sample)
            out[x_l: x_u] = sample

        # then, paramsets:
        for param_set in self.CP.get_parameter_sets():
            for i in range(self.num_samples):
                sample_dict = self.sample_iter_bound_map[i]
                x = param_set.get_iterate()
                (x_l, x_u) = sample_dict[x]
                sample = param_set.sample_point()
                # print(sample)
                out[x_l: x_u] = sample

        # then, steps:
        for i in range(self.num_samples):
            steps = self.alg_steps
            sample_dict = self.sample_iter_bound_map[i]
            for step in steps:
                for k in range(1, self.K + 1):
                    y_out = step.apply(k, self.iterate_to_id_map, sample_dict, out)
                    # (self, k, iter_to_id_map, ranges, out):
                    y_range = sample_dict[k][step.get_output_var()]
                    if y_out is not None:
                        out[y_range[0]: y_range[1]] = y_out

        # print(out, out.shape)
        return out

    def warmstartX(self):
        X_vec = self.warmstartX_vec()
        return X_vec @ X_vec.T

    def compare_warmstart(self):
        # start = time.time()
        X_resids, y_resids, feas_vals, obj_vals, cp_res = self.solve(
            plot=False, get_X=False, warmstart=False, return_resids=True)
        # end = time.time()
        ws_X, ws_y, ws_feas, ws_obj, _ = self.solve(plot=False, get_X=False, warmstart=True, return_resids=True)

        # fig, (ax0, ax1) = plt.subplots(2, figsize=(6, 8))
        fig, ax = plt.subplots(ncols=2, nrows=2, figsize=(10, 8))

        # fig.suptitle(f'CGAL progress, $K=1$, total time = {np.round(end-start, 3)} (s)')
        fig.suptitle('CGAL progress, 1')
        T = len(X_resids)
        # ax0.plot(range(1, T+1), X_resids, color='b', label='X resid')
        ax[0, 0].plot(range(1, T+1), X_resids, label='X_resid')
        ax[0, 0].set_yscale('log')
        ax[0, 0].set_ylabel('X_resids')
        ax[0, 0].plot(range(1, T+1), ws_X, linestyle='--')

        ax[0, 1].plot(range(1, T+1), y_resids, label='y_resid')
        ax[0, 1].set_yscale('log')
        ax[0, 1].set_ylabel('y_resids')
        ax[0, 1].plot(range(1, T+1), ws_y, linestyle='--')

        ax[1, 0].plot(range(1, T+1), feas_vals, label='feas dist')
        ax[1, 0].set_yscale('log')
        ax[1, 0].set_ylabel('feas dist')
        ax[1, 0].plot(range(1, T+1), ws_feas, linestyle='--')

        ax[1, 1].plot(range(1, T+1), obj_vals, label='obj')
        ax[1, 1].plot(range(1, T+1), ws_obj, linestyle='--')
        ax[1, 1].axhline(y=cp_res, linestyle='--', color='black')
        ax[1, 1].set_ylabel('obj')
        ax[1, 1].set_yscale('symlog')
        # ax0.plot(range(1, T+1), y_resids, color='r', label='y resid')
        # ax0.plot(range(1, T+1), feas_vals, color='g', label='dist onto K')

        # ax0.plot(range(1, T+1), ws_X, color='b', linestyle='--')
        # ax0.plot(range(1, T+1), ws_y, color='r', linestyle='--')
        # ax0.plot(range(1, T+1), ws_feas, color='g', linestyle='--')
        # ax0.set_yscale('log')
        # ax0.legend()

        # ax1.plot(range(1, T+1), obj_vals, label='obj')
        # ax1.plot(range(1, T+1), ws_obj, linestyle='--')
        # ax1.axhline(y=cp_res, linestyle='--', color='black')
        # # ax.axhline(y=0, color='black')
        # # plt.title('Objectives')
        # plt.xlabel('$t$')
        # ax1.set_yscale('symlog')
        # ax1.legend()
        plt.legend()
        plt.show()

    def solve(self, plot=True, get_X=False, warmstart=False, return_resids=False, scale_alpha=False, **kwargs):
        cp_res = self.test_with_cvxpy()
        print('cp res:', cp_res)
        print('C_scale:', self.C_scale)
        # cp_res = 0

        if scale_alpha:
            # alpha_mul = 7
            alpha_mul = 9000
            alpha = 1
            # print(self.A_norms)
            # print(self.b_lowerbounds)
            # print(self.b_upperbounds)
            self.divide_b_arrays_scalar(alpha_mul)
            # print(self.b_lowerbounds)
            # print(self.b_upperbounds)
        else:
            # alpha = 140
            alpha = 9000
            alpha_mul = 1
            # alpha = 1000

        beta_zero = 1
        A_op = self.estimate_A_op()
        # exit(0)
        K = np.inf
        n = self.problem_dim
        d = len(self.A_matrices)
        if warmstart:
            print('warm starting')
            X = self.warmstartX()
            y = self.AX(X)
            # y = np.zeros(d)
            # exit(0)
            print('ws first obj:', np.trace(self.C_matrix @ X))
            print('ws proj dist:', self.proj_dist(self.AX(X)))
        else:
            X = np.zeros((n, n))
            y = np.zeros(d)
            print('non ws first obj:', np.trace(self.C_matrix @ X))
            print('non ws proj dist:', self.proj_dist(self.AX(X)))
        T = 2000
        obj_vals = []
        X_resids = []
        y_resids = []
        feas_vals = []

        # for debugging lanczos:
        xi_diffs = []
        v_norm_diffs = []

        # start = time.time()
        num_gamma_exceeds = 0
        for t in trange(1, T+1):
            # print(t)
            beta = beta_zero * np.sqrt(t + 1)
            eta = 2 / (t + 1)
            w = self.proj(self.AX(X) + y / beta)
            D_mat = self.C_matrix + self.Astar_z(y + beta * (self.AX(X) - w))
            # D_jax = jnp.asarray(D_mat.todense())
            # D_jax = jspa.BCOO.fromdense(D_mat.todense())
            # print(type(D_jax))
            # exit(0)

            def mv(v):
                return D_mat @ v

            # def jax_mv(v):
            #     return D_jax @ v

            # D = self.C_matrix + self.Astar_z(y + beta * (self.AX(X) - w))
            D = spa.linalg.LinearOperator((n, n), matvec=mv)
            # print(D)

            xi, v = self.minimum_eigvec(D)
            xi = np.real(xi[0])
            v = np.real(v)

            # qt = int(np.ceil((t ** .25) * np.log(n)))
            # qt = n
            # test_xi, test_v = self.lanczos(D, qt)
            # xi, v = self.lanczos(D, qt)
            # xi_diffs.append(np.abs(xi - test_xi))
            # v_norm_diffs.append(np.linalg.norm(v - test_v))

            # xi, test_v = self.lanczos_jax(mv, qt)
            # xi, v = self.lanczos_jax(jax_mv, qt)

            # print(np.linalg.norm(v-test_v))
            # print(xi)
            # exit(0)

            if xi < 0:
                H = alpha * np.outer(v, v)
            else:
                H = np.zeros(X.shape)

            # H = alpha * np.outer(v, v)
            # print(np.trace(H))

            Xnew = (1 - eta) * X + eta * H
            Xresid = np.linalg.norm(X - Xnew)
            # print('X resid:', Xresid)
            X = Xnew

            beta_plus = beta_zero * np.sqrt(t + 2)

            # compute gamma
            wbar = self.proj(self.AX(X) + y / beta_plus)
            # rhs = (2 * alpha * A_op) ** 2 * beta / (t + 1) ** 1.5
            rhs = 4 * beta * (eta ** 2) * (alpha ** 2) * (A_op ** 2)
            # print(alpha, A_op, beta, t)
            # print(rhs)
            rhs_denom = np.linalg.norm(self.AX(X) - wbar) ** 2
            gamma = rhs / rhs_denom
            if gamma >= beta_zero:
                # print('rhs:', rhs, 'rhs_denom:', rhs_denom, 'gamma:', gamma, 'scaled down:', gamma/(alpha ** 2))
                # print(gamma)
                # gamma /= (alpha ** 2)
                num_gamma_exceeds += 1
            gamma = min(beta_zero, gamma)
            # gamma = .01
            # print(gamma)
            # gamma = .01

            ynew = y + gamma * (self.AX(X) - wbar)
            yresid = np.linalg.norm(y-ynew)
            # print('y resid:', yresid)
            if np.linalg.norm(ynew) < K:
                y = ynew
            else:
                print('exceed')
            new_obj = alpha_mul * self.C_scale * np.trace(self.C_matrix @ X)
            # print('obj', new_obj)
            # obj_vals.append(np.trace(self.C_matrix @ X))
            obj_vals.append(new_obj)
            X_resids.append(Xresid)
            y_resids.append(yresid)
            # print('feas', self.proj_dist(self.AX(X)))
            feas_vals.append(self.proj_dist(self.AX(X)))
            # pt = np.trace(self.C_matrix @ X)
            # zt = self.AX(X)
            # print(pt + y @ wbar + .5 * beta * (zt - wbar) @ (zt + wbar) - xi)
            # print(np.linalg.norm(zt - wbar))
        # print(X_resids, len(X_resids))
        # end = time.time()
        print('number of times gamma exceeded beta_0:', num_gamma_exceeds)
        print('final obj:', new_obj)
        print(X[-1, -1])
        # exit(0)
        if plot:
            fig, (ax0, ax1) = plt.subplots(2, figsize=(6, 8))

            # fig.suptitle(f'CGAL progress, $K=1$, total time = {np.round(end-start, 3)} (s)')
            fig.suptitle('CGAL progress')
            ax0.plot(range(1, T+1), X_resids, label='X resid')
            ax0.plot(range(1, T+1), y_resids, label='y resid')
            ax0.plot(range(1, T+1), feas_vals, label='dist onto K')
            ax0.set_yscale('log')
            ax0.legend()

            ax1.plot(range(1, T+1), obj_vals, label='obj')
            ax1.axhline(y=cp_res, linestyle='--', color='black')
            # ax1.set_ylim(bottom=-1)
            # ax.axhline(y=0, color='black')
            # plt.title('Objectives')
            plt.xlabel('$t$')
            ax1.set_yscale('symlog')
            ax1.legend()
            # plt.show()
            plt.savefig('Figure1.pdf')
        # plt.savefig('test.pdf')
        if get_X:
            return X
        if return_resids:
            return X_resids, y_resids, feas_vals, obj_vals, cp_res
        return D, xi_diffs, v_norm_diffs

    def solve_sketchy(self, R=2):
        cp_res = self.test_with_cvxpy()
        print(cp_res)
        # cp_res = 0
        # exit(0)
        # np.random.seed(0)
        alpha = 10
        beta_zero = 1
        A_op = self.estimate_A_op()
        K = np.inf
        n = self.problem_dim
        d = len(self.A_matrices)
        z = np.zeros(d)
        y = np.zeros(d)
        S = NymstromSketch(n, R)
        T = 500
        obj_vals = []
        z_resids = []
        y_resids = []
        feas_vals = []
        start = time.time()
        for t in trange(1, T+1):
            beta = beta_zero * np.sqrt(t + 1)
            eta = 2 / (t + 1)
            # X = np.outer(z, z)
            w = self.proj(z + y / beta)
            D_mat = self.C_matrix + self.Astar_z(y + beta * (z - w))

            def mv(v):
                return D_mat @ v

            # D = self.C_matrix + self.Astar_z(y + beta * (z - w))
            D = spa.linalg.LinearOperator((n, n), matvec=mv)

            xi, v = self.minimum_eigvec(D)
            xi = np.real(xi[0])
            v = np.real(v)

            # qt = int(np.ceil((t ** .25) * np.log(n)))
            # print(qt)
            # xi, test_v = self.lanczos(D, qt)
            # xi, v = self.lanczos(D, qt)
            # print(v.shape, test_v.shape)
            # print(xi)
            # print(np.abs(xi - test_xi))
            # print(np.linalg.norm(v - test_v))
            # print(test_xi - test_v.T @ D @ test_v)
            # print((test_v.T @ D @ test_v)[0][0] - xi)

            if xi < 0:
                H = alpha * np.outer(v, v)
                znew = (1 - eta) * z + eta * self.AX(H)
            else:
                # H = np.zeros((v.shape[0], v.shape[0]))
                znew = (1 - eta) * z

            # znew = (1 - eta) * z + eta * self.AX(H)
            zresid = np.linalg.norm(z - znew)
            z = znew

            beta_plus = beta_zero * np.sqrt(t + 2)

        #     # compute gamma
            wbar = self.proj(z + y / beta_plus)
        #     # rhs = (2 * alpha * A_op) ** 2 * beta / (t + 1) ** 1.5
            # rhs = 4 * beta * (eta ** 2) * (alpha ** 2) * (A_op ** 2)
            rhs = beta * (eta ** 2) * (alpha ** 2) * (A_op ** 2)
        #     # print(alpha, A_op, beta, t)
        #     # print(rhs)
            gamma = rhs / np.linalg.norm(z - wbar) ** 2
            gamma = min(beta_zero, gamma)
        #     # gamma = .01
        #     # print(gamma)
        #     # gamma = .01

            ynew = y + gamma * (z - wbar)
            yresid = np.linalg.norm(y-ynew)
        #     # print('y resid:', yresid)
            if np.linalg.norm(ynew) < K:
                y = ynew
            else:
                print('exceed')
        #     new_obj = np.trace(self.C_matrix @ X)
        #     # print('obj', new_obj)
        #     # obj_vals.append(np.trace(self.C_matrix @ X))
        #     obj_vals.append(new_obj)
        #     X_resids.append(Xresid)
            z_resids.append(zresid)
            y_resids.append(yresid)
        #     # print('feas', self.proj_dist(self.AX(X)))
            feas_vals.append(self.proj_dist(z))
        #     # pt = np.trace(self.C_matrix @ X)
        #     # zt = self.AX(X)
        #     # print(pt + y @ wbar + .5 * beta * (zt - wbar) @ (zt + wbar) - xi)
        #     # print(np.linalg.norm(zt - wbar))
            S.rank_one_update(np.sqrt(alpha) * v, eta)
            U, Delta = S.reconstruct()
            p = np.trace(self.C_matrix @ U @ np.diag(Delta) @ U.T)
            obj_vals.append(p)
        # # print(X_resids, len(X_resids))
        end = time.time()
        print(end-start)
        fig, (ax0, ax1) = plt.subplots(2, figsize=(6, 8))

        fig.suptitle(f'SketchyCGAL progress, $R={R}$, $K=1$, total time = {np.round(end-start, 3)} (s)')
        ax0.plot(range(1, T+1), z_resids, label='z resid')
        ax0.plot(range(1, T+1), y_resids, label='y resid')
        ax0.plot(range(1, T+1), feas_vals, label='dist onto K')
        ax0.set_yscale('log')
        ax0.legend()

        ax1.plot(range(1, T+1), obj_vals, label='obj')
        ax1.axhline(y=cp_res, linestyle='--', color='black')
        # ax.axhline(y=0, color='black')
        # plt.title('Objectives')
        plt.xlabel('$t$')
        ax1.set_yscale('symlog')
        ax1.legend()
        plt.show()
        # # plt.savefig('test.pdf')
        # return 0
