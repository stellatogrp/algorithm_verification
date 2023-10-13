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
from algocert.solvers.sdp_custom_solver.cross_constraints import (
    cross_constraints_between_linsteps,
    cross_constraints_linstep_to_not,
)
from algocert.solvers.sdp_custom_solver.RLT_constraints import RLT_all_vars, RLT_diagonal_vars
from algocert.solvers.sdp_custom_solver.solve_via_custom_admm import solve_via_custom_admm
from algocert.solvers.sdp_custom_solver.solve_via_mosek import solve_via_mosek
from algocert.solvers.sdp_custom_solver.solve_via_scs import solve_via_scs


class SDPCustomHandler(object):

    def __init__(self, CP, **kwargs):
        self.CP = CP
        self.K = CP.K
        self.alg_steps = CP.get_algorithm_steps()
        self.iterate_list = []
        self.param_list = []
        self.linstep_output_vars = []
        self.var_linstep_map = {}
        self.iterate_to_id_map = {}
        self.id_to_iterate_map = {}
        self.iter_bound_map = {}
        self.param_bound_map = {}
        self.iterate_init_set_map = {}
        self.param_set_map = {}
        self.range_marker = 0
        self.problem_dim = 0
        self.A_matrices = []
        self.A_norms = []
        self.b_lowerbounds = []
        self.b_upperbounds = []
        self.psd_cone_handlers = []
        self.C_matrix = None

        self._process_kwargs(**kwargs)

    def _process_kwargs(self, **kwargs):
        if 'scale' in kwargs:
            self.scale = kwargs['scale']
        else:
            self.scale = False

        if 'add_RLT' in kwargs:
            self.add_RLT = kwargs['add_RLT']
        else:
            self.add_RLT = True

        if 'add_RLT_diag' in kwargs:
            self.add_RLT_diag = kwargs['add_RLT_diag']
        else:
            self.add_RLT_diag = False

        if 'add_indiv_RLT' in kwargs:
            self.add_indiv_RLT = kwargs['add_indiv_RLT']
        else:
            self.add_indiv_RLT = False

        if 'add_planet' in kwargs:
            self.add_planet = kwargs['add_planet']
        else:
            self.add_planet = True

        if 'lookback_t' in kwargs:
            self.lookback_t = kwargs['lookback_t']
        else:
            self.lookback_t = None

        if 'couple_single_psd_cone' in kwargs:
            self.couple_single_psd_cone = kwargs['couple_single_psd_cone']
        else:
            self.couple_single_psd_cone = False

        if 'use_holder' in kwargs:
            self.use_holder = kwargs['use_holder']
        else:
            self.use_holder = True

        if 'sdp_solver' in kwargs:
            self.sdp_solver = kwargs['sdp_solver']
        else:
            # self.sdp_solver = 'scs'
            self.sdp_solver = 'mosek'

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
            self.param_list.append(param_var)
            dim = param_var.get_dim()
            bounds = (self.range_marker, self.range_marker + dim)
            self.range_marker += dim
            self.param_bound_map[param_var] = bounds
            self.param_set_map[param_var] = param_set
        print(self.param_bound_map)

    # def create_param_set_holder_map(self):
    #     pass

    def create_iterate_range_maps(self):
        for init_set in self.CP.get_init_sets():
            init_iter = init_set.get_iterate()
            dim = init_iter.get_dim()
            bounds = (self.range_marker, self.range_marker + dim)
            self.range_marker += dim
            self.iter_bound_map[init_iter] = {0: bounds}
            self.iterate_init_set_map[init_iter] = init_set

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
        self.var_warmstart = np.zeros((self.problem_dim - 1, 1))

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

    def extract_lin_step_output_vars(self):
        steps = self.CP.get_algorithm_steps()
        for step in steps:
            if step.is_linstep:
                # self.linstep_output_vars.add(step.get_output_var())
                self.linstep_output_vars.append(step.get_output_var())
                self.var_linstep_map[step.get_output_var()] = step
        # print(self.linstep_output_vars)
        # print(self.var_linstep_map)

    def create_lower_right_constraint(self):
        A = spa.lil_matrix((self.problem_dim, self.problem_dim))
        # A = spa.csc_matrix((self.problem_dim, self.problem_dim))
        A[-1, -1] = 1
        self.A_matrices.append(A.tocsc())
        self.b_lowerbounds.append(1)
        self.b_upperbounds.append(1)

    def add_RLT_constraints(self):
        mat_dim = self.problem_dim - 1
        # A, b_l, b_u = RLT_from_ranges(mat_range, mat_range, self)
        A, b_l, b_u = RLT_all_vars(mat_dim, self)
        self.A_matrices += A
        self.b_lowerbounds += b_l
        self.b_upperbounds += b_u

    def add_RLT_diag_constraints(self):
        mat_dim = self.problem_dim - 1
        A, b_l, b_u = RLT_diagonal_vars(mat_dim, self)
        self.A_matrices += A
        self.b_lowerbounds += b_l
        self.b_upperbounds += b_u

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
            A, b_l, b_u, psd_cones = canon_method(init_set, self)
            self.A_matrices += A
            self.b_lowerbounds += b_l
            self.b_upperbounds += b_u
            self.psd_cone_handlers += psd_cones

    def canonicalize_parameter_sets(self):
        for param_set in self.CP.get_parameter_sets():
            # print(param_set)
            canon_method = SET_CANON_METHODS[type(param_set)]
            A, b_l, b_u, psd_cones = canon_method(param_set, self)
            self.A_matrices += A
            self.b_lowerbounds += b_l
            self.b_upperbounds += b_u
            self.psd_cone_handlers += psd_cones

    def canonicalize_steps(self):
        steps = self.CP.get_algorithm_steps()
        for k in range(1, self.K + 1):
            for step in steps:
                canon_method = STEP_CANON_METHODS[type(step)]
                A, b_l, b_u, psd_cones = canon_method(step, k, self)
                self.A_matrices += A
                self.b_lowerbounds += b_l
                self.b_upperbounds += b_u
                self.psd_cone_handlers += psd_cones

    def canonicalize_linstep_cross_constraints(self):
        # print(self.linstep_output_vars)
        # print(self.var_linstep_map)
        # print(self.iterate_list)
        # print(self.param_list)
        if self.lookback_t is not None:
            lookback_t = self.lookback_t
        else:
            lookback_t = self.K
            # lookback_t = 1
        # print('lookback_t:', lookback_t)
        # print(len(self.A_matrices), len(self.b_lowerbounds), len(self.b_upperbounds))
        for linstep_var in self.linstep_output_vars:
            for other_var in self.iterate_list:
                for k1 in range(1, self.K + 1):
                    for k2 in range(k1, max(0, k1 - lookback_t) - 1, -1):
                        if linstep_var == other_var and k1 == k2:
                            continue
                        # print(linstep_var, other_var, 'k1:', k1, 'k2:', k2)
                        if other_var in self.linstep_output_vars:
                            # print('other also linstep')
                            A, b_l, b_u, psd_cones = cross_constraints_between_linsteps(linstep_var, other_var, k1, k2, self,
                                                                                        only_include_psd_cones=True)
                            # self.A_matrices += A
                            # self.b_lowerbounds += b_l
                            # self.b_upperbounds += b_u
                            self.psd_cone_handlers += psd_cones
                        else:
                            # print('other is not linstep')
                            A, b_l, b_u, psd_cones = cross_constraints_linstep_to_not(linstep_var, other_var, k1, k2, self)
                            self.A_matrices += A
                            self.b_lowerbounds += b_l
                            self.b_upperbounds += b_u
                            self.psd_cone_handlers += psd_cones
        # TODO: add linsteps cross with params?
        # for linstep_var in self.linstep_output_vars:
        #     for param_var in self.param_list:
        #         for k in range(1, self.K + 1):
        #             A, b_l, b_u, psd_cones = cross_constraints_linstep_to_not(linstep_var, param_var, k, None, self)
        #             self.A_matrices += A
        #             self.b_lowerbounds += b_l
        #             self.b_upperbounds += b_u
                    # self.psd_cone_handlers += psd_cones

        # print(len(self.A_matrices), len(self.b_lowerbounds), len(self.b_upperbounds))
        # exit(0)

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

        self.extract_lin_step_output_vars()

        self.create_lower_right_constraint()

        if self.add_RLT:
            self.add_RLT_constraints()

        if self.add_RLT_diag:
            self.add_RLT_diag_constraints()

        self.canonicalize_objective()
        self.canonicalize_initial_sets()
        self.canonicalize_parameter_sets()

        self.canonicalize_steps()

        self.canonicalize_linstep_cross_constraints()

        self.check_all_matrices_symmetric()
        # self.solve_with_cvxpy()

    def solve_with_cvxpy(self):
        # print(len(self.A_matrices), len(self.b_lowerbounds), len(self.b_upperbounds))
        X = cp.Variable((self.problem_dim, self.problem_dim), symmetric=True)
        obj = cp.trace(self.C_matrix @ X)
        constraints = []
        if self.couple_single_psd_cone:
            constraints += [X >> 0]
            # Z = cp.Variable((self.problem_dim, self.problem_dim), symmetric=True)
            # ranges = (0, self.problem_dim - 1)
            # h = PSDConeHandler(ranges)
            # E = h.get_E_mat(self.problem_dim)
            # constraints = [Z == E @ X @ E.T, Z >> 0]
        else:
            for h in self.psd_cone_handlers:
                Z = cp.Variable((h.ranges_dim, h.ranges_dim), symmetric=True)
                E = spa.csc_matrix(h.get_E_mat(self.problem_dim))
                # print(E)
                constraints += [Z == E @ X @ E.T, Z >> 0]
            # constraints = [X >> 0]
            # exit(0)
        for i in range(len(self.A_matrices)):
            Ai = self.A_matrices[i]
            b_li = self.b_lowerbounds[i]
            b_ui = self.b_upperbounds[i]
            # if b_li == -np.inf:
            #     b_li = -500
            # if b_ui == np.inf:
            #     b_ui = 500

            if b_li == b_ui:
                constraints += [cp.trace(Ai @ X) == b_li]
            else:
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

    def solve_with_admm(self):
        out = solve_via_custom_admm(self.C_matrix, self.A_matrices, self.b_lowerbounds, self.b_upperbounds,
                                    self.psd_cone_handlers, self.problem_dim)
        return out

    def solve_with_scs_directly(self):
        # int(self.problem_dim * (self.problem_dim + 1) / 2)
        out = solve_via_scs(self.C_matrix, self.A_matrices, self.b_lowerbounds, self.b_upperbounds,
                            self.psd_cone_handlers, self.problem_dim, self)
        return out

    def solve_with_mosek_directly(self):
        out = solve_via_mosek(self.C_matrix, self.A_matrices, self.b_lowerbounds, self.b_upperbounds,
                              self.psd_cone_handlers, self.problem_dim)
        return out

    def solve(self):
        # return self.solve_with_cvxpy()
        # return self.solve_with_scs_directly()
        if self.sdp_solver == 'mosek':
            return self.solve_with_mosek_directly()
        if self.sdp_solver == 'scs':
            return self.solve_with_scs_directly()
        # return self.solve_with_admm()
