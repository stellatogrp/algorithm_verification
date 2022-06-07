import cvxpy as cp


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
        self.iteration_handlers = []
        self.sdp_constraints = []
        self.sdp_obj = 0

    def create_iterate_id_maps(self):
        steps = self.CP.get_algorithm_steps()
        counter = 0
        for step in steps:
            iterate = step.get_output_var()
            self.iterate_to_id_map[iterate] = counter
            self.id_to_iterate_map[counter] = iterate
            self.iterate_list.append(iterate)
            counter += 1
        # print(self.iterate_list)

    def compute_sdp_param_vars(self):
        param_sets = self.CP.get_parameter_sets()

        for p in param_sets:
            param_var = p.get_iterate()
            n = param_var.get_dim()
            self.sdp_param_vars[param_var] = cp.Variable((n, 1))
            self.sdp_param_outerproduct_vars[param_var] = cp.Variable((n, n))
            self.param_list.append(param_var)
        # print(self.sdp_param_outerproduct_vars)
        # print(self.param_list)

    def create_iteration_handlers(self):
        steps = self.CP.get_algorithm_steps()
        for k in range(self.N + 1):
            self.iteration_handlers.append(SingleIterationHandler(k, steps, self.iterate_list, self.param_list))

    def canonicalize_initial_sets(self):
        pass

    def canonicalize_parameter_sets(self):
        pass

    def canonicalize_steps(self):
        pass

    def canonicalize_obejective(self):
        pass

    def canonicalize(self):
        self.create_iterate_id_maps()
        self.compute_sdp_param_vars()
        self.create_iteration_handlers()
        self.canonicalize_steps()


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
            self.iterate_vars[iterate] = cp.Variable((n, 1))
            self.iterate_outerproduct_vars[iterate] = cp.Variable((n, n))

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
            for iter2 in self.iterates:
                m = iter2.get_dim()
                cross_dict[iter2] = cp.Variable((n, m))
            self.iterate_cross_vars[iter1] = cross_dict
            # print(iter1, self.iterate_cross_vars[iter1])
