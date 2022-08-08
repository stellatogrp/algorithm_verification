import numpy as np
import cvxpy as cp

from certification_problem.solvers.sdp_solver.obj_canonicalizer.convergence_residual import conv_resid_canon
from certification_problem.solvers.sdp_solver.var_bounds.var_bounds import CPVarAndBounds
from certification_problem.solvers.sdp_solver import SET_CANON_METHODS, STEP_CANON_METHODS


class SDPHandler(object):

    def __init__(self, CP):
        self.CP = CP
        self.N = self.CP.N
        self.iterate_list = []
        self.param_list = []
        self.sdp_param_vars = {}
        self.sdp_param_outerproduct_vars = {}
        self.iterate_to_id_map = {}
        self.id_to_iterate_map = {}
        self.iterate_to_type_map = {}
        self.iteration_handlers = []
        self.sdp_constraints = []
        self.sdp_obj = 0

    def create_iterate_id_maps(self):
        steps = self.CP.get_algorithm_steps()
        for i, step in enumerate(steps):
            iterate = step.get_output_var()
            self.iterate_to_id_map[iterate] = i
            self.id_to_iterate_map[i] = iterate
            self.iterate_list.append(iterate)

    def compute_sdp_param_vars(self):
        param_sets = self.CP.get_parameter_sets()

        for p in param_sets:
            param_var = p.get_iterate()
            n = param_var.get_dim()
            self.sdp_param_vars[param_var] = CPVarAndBounds((n, 1))
            self.sdp_param_outerproduct_vars[param_var] = cp.Variable((n, n), symmetric=True)
            self.param_list.append(param_var)
        # print(self.sdp_param_outerproduct_vars)
        # print(self.param_list)

    def create_iteration_handlers(self):
        steps = self.CP.get_algorithm_steps()
        for k in range(self.N + 1):
            self.iteration_handlers.append(SingleIterationHandler(k, steps, self.iterate_list, self.param_list))

    def canonicalize_initial_sets(self):
        for init_set in self.CP.get_init_sets():
            canon_method = SET_CANON_METHODS[type(init_set)]
            # constraints = l2_ball_canon(init_set, self.iteration_handlers[0])
            constraints = canon_method(init_set, self.iteration_handlers[0])
            self.sdp_constraints += constraints

    def canonicalize_parameter_sets(self):
        # TODO add other sets
        # TODO add cross terms with multiple params
        for param_set in self.CP.get_parameter_sets():
            # r = param_set.r
            # param = param_set.get_iterate()
            # b = self.sdp_param_vars[param]
            # bbT = self.sdp_param_outerproduct_vars[param]
            # self.sdp_constraints += [
            #     cp.sum_squares(b) <= r ** 2, cp.trace(bbT) <= r ** 2,
            # ]

            # NOTE this is a massive placeholder
            param_handler = ParameterHandler(self.sdp_param_vars, self.sdp_param_outerproduct_vars)
            canon_method = SET_CANON_METHODS[type(param_set)]
            constraints = canon_method(param_set, param_handler)
            self.sdp_constraints += constraints

    def canonicalize_steps(self):
        steps = self.CP.get_algorithm_steps()
        for k in range(1, self.N + 1):
            curr = self.iteration_handlers[k]
            prev = self.iteration_handlers[k - 1]
            for i, step in enumerate(steps):
                prev_step = steps[i-1]
                output_var = step.get_output_var()
                self.iterate_to_type_map[output_var] = type(step)
                canon_method = STEP_CANON_METHODS[type(step)]
                constraints = canon_method(steps, i, curr, prev, self.iterate_to_id_map,
                                           self.sdp_param_vars, self.sdp_param_outerproduct_vars)
                self.sdp_constraints += constraints

    def add_convexity_constraints(self, A):
        # TODO add this as a settable flag
        x = self.iterate_list[-1]  # TODO placeholder until function allows iterate specification
        b = self.param_list[-1]

        for k in range(1, self.N + 1):
            handler_k = self.iteration_handlers[k]
            handler_kminus1 = self.iteration_handlers[k-1]
            xk = handler_k.iterate_vars[x]
            xkxkT = handler_k.iterate_outerproduct_vars[x]
            xkminus1 = handler_kminus1.iterate_vars[x]
            xkminus1_xkminus1T = handler_kminus1.iterate_outerproduct_vars[x]
            xk_xkminus1T = handler_k.iterate_cross_vars[x][x]
            self.sdp_constraints += [
                cp.trace(A @ xkxkT) + cp.trace(A @ xkminus1_xkminus1T) >= 2 * cp.trace(A @ xk_xkminus1T),
                # cp.bmat([
                #     [xkxkT, xk_xkminus1T, xk],
                #     [xk_xkminus1T.T, xkminus1_xkminus1T, xkminus1],
                #     [xk.T, xkminus1.T, np.array([[1]])]
                # ]) >> 0,
                # cp.trace(A @ xkxkT) + cp.trace(A @ bbT_var) >= 2 * cp.trace(A @ xkbT)
            ]

    def canonicalize_objective(self):
        obj = self.CP.objective
        # print(obj)
        sdp_obj, constraints = conv_resid_canon(obj,
                                                self.iteration_handlers[self.N], self.iteration_handlers[self.N - 1])
        self.sdp_obj += sdp_obj
        self.sdp_constraints += constraints

    def canonicalize(self):
        self.create_iterate_id_maps()
        self.compute_sdp_param_vars()
        self.create_iteration_handlers()
        self.canonicalize_initial_sets()
        self.canonicalize_parameter_sets()
        self.canonicalize_steps()
        # self.add_convexity_constraints()  # TODO add this as a settable flag
        self.canonicalize_objective()
        # print(len(self.sdp_constraints))

    def solve(self):
        prob = cp.Problem(cp.Maximize(self.sdp_obj), self.sdp_constraints)
        res = prob.solve()
        print(res)


class SingleIterationHandler(object):

    def __init__(self, k, steps, iterates, parameters):
        self.k = k
        self.steps = steps
        self.iterates = iterates
        self.parameters = parameters
        self.iterate_vars = {}
        self.iterate_outerproduct_vars = {}
        self.iterate_cross_vars = {}
        self.iterate_param_vars = {}

        self.create_noncross_iterate_vars()
        self.create_iterate_param_vars()
        self.create_iterate_cross_vars()

    def create_noncross_iterate_vars(self):
        for iterate in self.iterates:
            n = iterate.get_dim()
            self.iterate_vars[iterate] = CPVarAndBounds((n, 1))
            self.iterate_outerproduct_vars[iterate] = cp.Variable((n, n), symmetric=True)

    def create_iterate_param_vars(self):
        for iterate in self.iterates:
            n = iterate.get_dim()
            param_dict = {}
            for param in self.parameters:
                m = param.get_dim()
                param_dict[param] = cp.Variable((n, m))
            self.iterate_param_vars[iterate] = param_dict

    def create_iterate_cross_vars(self):
        for iter1 in self.iterates:
            n = iter1.get_dim()
            cross_dict = {}
            # Note that the same iterates are also included because
            # self.iterate_cross_vars[x][x] -> (x^k)(x^{k+1})^T (the previous iteration)
            # To get (x^k)(x^K)^T, use self.iterate_outerproduct_vars
            for iter2 in self.iterates:
                m = iter2.get_dim()
                cross_dict[iter2] = cp.Variable((n, m))
            self.iterate_cross_vars[iter1] = cross_dict
            # print(iter1, self.iterate_cross_vars[iter1])


class ParameterHandler(object):
    # TODO this is just a massive placeholder to overlap the parameter/initial iterate in canon methods
    def __init__(self, param_vars, param_outerproduct_vars):
        self.iterate_vars = param_vars
        self.iterate_outerproduct_vars = param_outerproduct_vars
