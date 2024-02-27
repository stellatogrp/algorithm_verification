import cvxpy as cp

from algoverify.solvers.sdp_solver import (
    HL_TO_BASIC_STEP_METHODS,
    OBJ_CANON_METHODS,
    RLT_CANON_SET_METHODS,
    RLT_CANON_STEP_METHODS,
    SET_CANON_METHODS,
    STEP_CANON_METHODS,
)
from algoverify.solvers.sdp_solver.var_bounds.var_bounds import CPVarAndBounds


class SDPHandler(object):

    def __init__(self, CP, **kwargs):
        self.CP = CP
        self.K = self.CP.K
        self.iterate_list = []
        self.param_list = []
        self.sdp_param_vars = {}
        self.sdp_param_outerproduct_vars = {}
        self.iterate_to_id_map = {}
        self.id_to_iterate_map = {}
        self.iterate_to_type_map = {}
        self.iteration_handlers = []
        self.linstep_output_vars = set([])
        self.var_linstep_map = {}
        self.sdp_constraints = []
        self.sdp_obj = 0

        if 'add_RLT' in kwargs:
            self.add_RLT = kwargs['add_RLT']
        else:
            self.add_RLT = False

        if 'add_planet' in kwargs:
            self.add_planet = kwargs['add_planet']
        else:
            self.add_planet = False

        if 'minimize' in kwargs:
            self.minimize = kwargs['minimize']
        else:
            self.minimize = False

        self.kwargs = kwargs

    def convert_hl_to_basic_steps(self):
        all_steps_canonicalizeable = True
        steps = self.CP.get_algorithm_steps()
        new_steps = []
        for step in steps:
            if type(step) in HL_TO_BASIC_STEP_METHODS:
                all_steps_canonicalizeable = False
                canon_method = HL_TO_BASIC_STEP_METHODS[type(step)]
                new_vars, canon_steps = canon_method(step)
                new_steps += canon_steps
            else:
                new_steps += [step]
        self.CP.set_algorithm_steps(new_steps)
        if not all_steps_canonicalizeable:
            self.convert_hl_to_basic_steps()

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
        for k in range(self.K + 1):
            self.iteration_handlers.append(SingleIterationHandler(k, steps, self.iterate_list, self.param_list))

    def extract_lin_step_output_vars(self):
        # print('extracting')
        steps = self.CP.get_algorithm_steps()
        for step in steps:
            # print(step.get_output_var())
            if step.is_linstep:
                self.linstep_output_vars.add(step.get_output_var())
                self.var_linstep_map[step.get_output_var()] = step
        # print(self.linstep_output_vars)

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
            # NOTE this is a massive placeholder
            param_handler = ParameterHandler(self.sdp_param_vars, self.sdp_param_outerproduct_vars)
            canon_method = SET_CANON_METHODS[type(param_set)]
            constraints = canon_method(param_set, param_handler)
            self.sdp_constraints += constraints

    def canonicalize_steps(self):
        steps = self.CP.get_algorithm_steps()
        for k in range(1, self.K + 1):
            self.iteration_handlers[k]
            self.iteration_handlers[k - 1]
            for i, step in enumerate(steps):
                output_var = step.get_output_var()
                self.iterate_to_type_map[output_var] = type(step)
                canon_method = STEP_CANON_METHODS[type(step)]
                # constraints = canon_method(steps, i, curr, prev, self.iterate_to_id_map,
                #                            self.sdp_param_vars, self.sdp_param_outerproduct_vars, self.add_RLT,
                #                            self.kwargs)
                constraints = canon_method(steps, i, self.iteration_handlers, k, self.iterate_to_id_map,
                                           self.sdp_param_vars, self.sdp_param_outerproduct_vars,
                                           self.var_linstep_map,
                                           self.add_RLT,
                                           self.kwargs)
                self.sdp_constraints += constraints

    def canonicalize_objective(self):
        # obj = self.CP.objective
        # obj_canon = OBJ_CANON_METHODS[type(obj)]
        # sdp_obj, constraints = obj_canon(obj, self.iteration_handlers, self.add_RLT)
        # self.sdp_obj += sdp_obj
        # self.sdp_constraints += constraints
        # if type(self.CP.objective) != list:
        if not isinstance(self.CP.objective, list):
            obj_list = [self.CP.objective]
        else:
            obj_list = self.CP.objective

        for obj in obj_list:
            obj_canon = OBJ_CANON_METHODS[type(obj)]
            single_obj, constraints = obj_canon(obj, self.iteration_handlers, self.add_RLT)
            self.sdp_obj += single_obj
            self.sdp_constraints += constraints

    def propagate_lower_upper_bounds(self):
        self.initialize_init_set_bounds()
        self.initialize_param_set_bounds()
        self.propagate_iterate_bounds()
        # for i in self.param_list:
        #     print('param', i)
        #     print(self.sdp_param_vars[i].get_lower_bound())
        # for i, handler in enumerate(self.iteration_handlers):
        #     print('handler', i)
        #     for iterate in self.iterate_list:
        #         # print(handler.iterate_vars[iterate])
        #         print(iterate)
        #         print(handler.iterate_vars[iterate].get_lower_bound())
        # exit(0)

    def initialize_init_set_bounds(self):
        for init_set in self.CP.get_init_sets():
            canon_method = RLT_CANON_SET_METHODS[type(init_set)]
            canon_method(init_set, self.iteration_handlers[0])
            # x = init_set.get_iterate()
            # print(self.iteration_handlers[0].iterate_vars[x].get_lower_bound())
            # print(self.iteration_handlers[0].iterate_vars[x].get_upper_bound())

    def initialize_param_set_bounds(self):
        for param_set in self.CP.get_parameter_sets():
            # NOTE this is a massive placeholder
            param_handler = ParameterHandler(self.sdp_param_vars, self.sdp_param_outerproduct_vars)
            canon_method = RLT_CANON_SET_METHODS[type(param_set)]
            canon_method(param_set, param_handler)
            # p = param_set.get_iterate()
            # print(p)
            # print(self.sdp_param_vars[p].get_lower_bound())
            # print(self.sdp_param_vars[p].get_upper_bound())

    def propagate_iterate_bounds(self):
        steps = self.CP.get_algorithm_steps()
        for k in range(1, self.K + 1):
            curr = self.iteration_handlers[k]
            prev = self.iteration_handlers[k - 1]
            for i, step in enumerate(steps):
                canon_method = RLT_CANON_STEP_METHODS[type(step)]
                canon_method(steps, i, curr, prev, self.iterate_to_id_map,
                             self.sdp_param_vars, self.sdp_param_outerproduct_vars)
                # u = step.get_output_var()
                # print(prev.iterate_vars[u].get_upper_bound())
                # print(curr.iterate_vars[u].get_upper_bound())

    def canonicalize(self):
        self.convert_hl_to_basic_steps()
        # print('----BASIC STEP SDP----')
        # self.CP.print_cp()

        self.create_iterate_id_maps()
        self.compute_sdp_param_vars()
        self.create_iteration_handlers()
        self.extract_lin_step_output_vars()

        if self.add_RLT:
            self.propagate_lower_upper_bounds()

        self.canonicalize_initial_sets()
        self.canonicalize_parameter_sets()
        self.canonicalize_steps()
        # self.add_convexity_constraints()  # TODO add this as a settable flag
        self.canonicalize_objective()
        # print(len(self.sdp_constraints))

    def solve(self, **kwargs):

        if self.minimize:
            obj = cp.Minimize(self.sdp_obj)
        else:
            obj = cp.Maximize(self.sdp_obj)

        prob = cp.Problem(obj, self.sdp_constraints)
        if 'solver' in kwargs:
            solver = kwargs['solver']
        else:
            solver = cp.MOSEK
        if 'verbose' in kwargs:
            verbose = kwargs['verbose']
        else:
            verbose = False
        res = prob.solve(solver=solver, verbose=verbose)
        # print(res)
        return res, prob.solver_stats.solve_time

    def get_iteration_handler(self, k):
        return self.iteration_handlers[k]


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
