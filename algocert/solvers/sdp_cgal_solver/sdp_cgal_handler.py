import time

import cvxpy as cp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
from tqdm import trange

from algocert.solvers.sdp_cgal_solver import (OBJ_CANON_METHODS,
                                              SET_CANON_METHODS,
                                              STEP_CANON_METHODS)


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
        self.b_lowerbounds = []
        self.b_upperbounds = []
        self.C_matrix = None

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

    def estimate_A_op(self):
        X = np.random.rand(self.problem_dim, self.problem_dim)
        X = X @ X.T
        X = X / np.linalg.norm(X)
        z = self.AX(X)
        return np.linalg.norm(z)

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

    def solve(self):
        cp_res = self.test_with_cvxpy()
        print(cp_res)
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
        start = time.time()
        for t in trange(1, T+1):
            # print(t)
            beta = beta_zero * np.sqrt(t + 1)
            eta = 2 / (t + 1)
            w = self.proj(self.AX(X) + y / beta)
            D = self.C_matrix + self.Astar_z(y + beta * (self.AX(X) - w))
            # print(D)

            # out = self.minimum_eigvec(D)
            xi, v = self.minimum_eigvec(D)
            xi = np.real(xi[0])
            v = np.real(v)

            # if xi > 0:
            #     H = alpha * np.outer(v, v)
            # else:
            #     H = np.zeros(X.shape)

            H = alpha * np.outer(v, v)
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
            gamma = rhs / np.linalg.norm(self.AX(X) - wbar) ** 2
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
            new_obj = np.trace(self.C_matrix @ X)
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
        end = time.time()
        fig, (ax0, ax1) = plt.subplots(2, figsize=(6, 8))

        fig.suptitle(f'CGAL progress, $K=1$, total time = {np.round(end-start, 3)} (s)')
        ax0.plot(range(1, T+1), X_resids, label='X resid')
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
        # plt.savefig('test.pdf')
        return 0
