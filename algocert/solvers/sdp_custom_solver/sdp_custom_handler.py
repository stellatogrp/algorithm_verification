import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver import (
    OBJ_CANON_METHODS,
    SET_BOUND_CANON_METHODS,
    SET_CANON_METHODS,
    STEP_BOUND_CANON_METHODS,
    STEP_CANON_METHODS,
)


class SDPCustomHandler(object):

    def __init__(self, CP, **kwargs):
        self.CP = CP
        self.K = CP.K
        self.alg_steps = CP.get_algorithm_steps()
        self.iterate_list = []
        self.param_list = []
        self.iterate_to_id_map = {}
        self.id_to_iterate_map = {}
        self.iter_bound_map = {}
        self.param_bound_map = {}
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

        if 'add_RLT' in kwargs:
            self.add_RLT = kwargs['add_RLT']
        else:
            self.add_RLT = True

        if 'add_planet' in kwargs:
            self.add_planet = kwargs['add_planet']
        else:
            self.add_planet = True

    def convert_hl_to_basic_steps(self):
        pass

    def create_iterate_id_maps(self):
        steps = self.CP.get_algorithm_steps()
        for i, step in enumerate(steps):
            iterate = step.get_output_var()
            self.iterate_to_id_map[iterate] = i
            self.id_to_iterate_map[i] = iterate
            self.iterate_list.append(iterate)
        print(self.iterate_to_id_map)

    # def create_init_iterate_range_maps(self):
    #     for init_set in self.CP.get_init_sets():
    #         init_iter = init_set.get_iterate()
    #         dim = init_iter.get_dim()
    #         bounds = (self.range_marker, self.range_marker + dim)
    #         self.range_marker += dim
    #         self.init_iter_range_map[init_iter] = bounds
    #     print(self.init_iter_range_map)

    def create_param_range_maps(self):
        for param_set in self.CP.get_parameter_sets():
            param_var = param_set.get_iterate()
            dim = param_var.get_dim()
            bounds = (self.range_marker, self.range_marker + dim)
            self.range_marker += dim
            self.param_bound_map[param_var] = bounds
        print(self.param_bound_map)

    def create_iterate_range_maps(self):
        for init_set in self.CP.get_init_sets():
            init_iter = init_set.get_iterate()
            dim = init_iter.get_dim()
            bounds = (self.range_marker, self.range_marker + dim)
            self.range_marker += dim
            self.iter_bound_map[init_iter] = {0: bounds}

        for k in range(1, self.K + 1):
            for step in self.CP.get_algorithm_steps():
                output_var = step.get_output_var()
                if output_var not in self.iter_bound_map:
                    self.iter_bound_map[output_var] = {}
                iter_dict = self.iter_bound_map[output_var]
                dim = output_var.get_dim()
                bounds = (self.range_marker, self.range_marker + dim)
                self.range_marker += dim
                iter_dict[k] = bounds

        print(self.iter_bound_map)

    def set_problem_dim(self):
        self.problem_dim = self.range_marker + 1
        print('problem dim:', self.problem_dim)

    def create_lower_upper_bound_vecs(self):
        self.var_lowerbounds = np.zeros((self.problem_dim - 1, 1))
        self.var_upperbounds = np.zeros((self.problem_dim - 1, 1))

    def initialize_set_bounds(self):
        # print('init')
        for init_set in self.CP.get_init_sets() + self.CP.get_parameter_sets():
            # print(init_set)
            canon_method = SET_BOUND_CANON_METHODS[type(init_set)]
            canon_method(init_set, self)
        # print(self.var_lowerbounds, self.var_upperbounds)

    def propagate_step_bounds(self):
        # print('step upper lower')
        for k in range(1, self.K + 1):
            for step in self.CP.get_algorithm_steps():
                # print(k, step)
                canon_method = STEP_BOUND_CANON_METHODS[type(step)]
                canon_method(step, k, self)
        # print(self.var_lowerbounds, self.var_upperbounds)
        # exit(0)

    def create_lower_right_constraint(self):
        A = spa.lil_matrix((self.problem_dim, self.problem_dim))
        # A = spa.csc_matrix((self.problem_dim, self.problem_dim))
        A[-1, -1] = 1
        self.A_matrices.append(A.tocsc())
        self.b_lowerbounds.append(1)
        self.b_upperbounds.append(1)

    def canonicalize_objective(self):
        # self.C_matrix = spa.lil_matrix((self.problem_dim, self.problem_dim))
        self.C_matrix = spa.csc_matrix((self.problem_dim, self.problem_dim))
        # if type(self.CP.objective) != list:
        if not isinstance(self.CP.objective, list):
            obj_list = [self.CP.objective]
        else:
            obj_list = self.CP.objective

        for obj in obj_list:
            obj_canon = OBJ_CANON_METHODS[type(obj)]
            C_temp = obj_canon(obj, self)
            # print(C_temp)
            self.C_matrix += C_temp

        # Flipping objective to make the problem a minimization
        self.C_matrix = -self.C_matrix

    def canonicalize_initial_sets(self):
        for init_set in self.CP.get_init_sets():
            # print(init_set)
            canon_method = SET_CANON_METHODS[type(init_set)]
            A, b_l, b_u = canon_method(init_set, self)
            self.A_matrices += A
            self.b_lowerbounds += b_l
            self.b_upperbounds += b_u

    def canonicalize_parameter_sets(self):
        for param_set in self.CP.get_parameter_sets():
            # print(param_set)
            canon_method = SET_CANON_METHODS[type(param_set)]
            A, b_l, b_u = canon_method(param_set, self)
            self.A_matrices += A
            self.b_lowerbounds += b_l
            self.b_upperbounds += b_u

    def canonicalize_steps(self):
        steps = self.CP.get_algorithm_steps()
        for k in range(1, self.K + 1):
            for step in steps:
                canon_method = STEP_CANON_METHODS[type(step)]
                A, b_l, b_u = canon_method(step, k, self)
                self.A_matrices += A
                self.b_lowerbounds += b_l
                self.b_upperbounds += b_u

    def single_mat_symmetric(self, M):
        return (abs(M - M.T) > 1e-7).nnz == 0

    def check_all_matrices_symmetric(self):
        for A in self.A_matrices:
            if not self.single_mat_symmetric(A):
                print('mat not symmetric')
                exit(0)
        else:
            print('all input matrices symmetric')

    def canonicalize(self):
        self.convert_hl_to_basic_steps()
        self.create_iterate_id_maps()
        # self.create_init_iterate_range_maps()
        self.create_param_range_maps()
        self.create_iterate_range_maps()
        self.set_problem_dim()
        self.create_lower_upper_bound_vecs()

        # if self.add_RLT:
        self.initialize_set_bounds()
        self.propagate_step_bounds()

        self.create_lower_right_constraint()

        self.canonicalize_objective()
        self.canonicalize_initial_sets()
        self.canonicalize_parameter_sets()

        self.canonicalize_steps()

        self.check_all_matrices_symmetric()
        # self.solve_with_cvxpy()

    def solve_with_cvxpy(self):
        # print(len(self.A_matrices), len(self.b_lowerbounds), len(self.b_upperbounds))
        X = cp.Variable((self.problem_dim, self.problem_dim), symmetric=True)
        obj = cp.trace(self.C_matrix @ X)
        constraints = [X >> 0]
        for i in range(len(self.A_matrices)):
            Ai = self.A_matrices[i]
            b_li = self.b_lowerbounds[i]
            b_ui = self.b_upperbounds[i]
            # if b_li == -np.inf:
            #     b_li = -500
            # if b_ui == np.inf:
            #     b_ui = 500

            if b_li > -np.inf:
                constraints += [cp.trace(Ai @ X) >= b_li]
            if b_ui < np.inf:
                constraints += [cp.trace(Ai @ X) <= b_ui]
            # constraints += [
            #     cp.trace(Ai @ X) >= b_li,
            #     cp.trace(Ai @ X) <= b_ui,
            # ]

        prob = cp.Problem(cp.Minimize(obj), constraints)
        res = prob.solve(solver=cp.MOSEK, verbose=True)
        # res = prob.solve(solver=cp.SCS, verbose=True, max_iters=10000)
        # print(res)
        return -res, prob.solver_stats.solve_time

    def solve(self):
        return self.solve_with_cvxpy()
