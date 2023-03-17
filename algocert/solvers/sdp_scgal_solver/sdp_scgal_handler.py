

class SDPSCGALHandler(object):

    def __init__(self, CP, **kwargs):
        self.CP = CP
        self.K = CP.K
        self.num_samples = CP.num_samples
        self.alg_steps = CP.get_algorithm_steps()
        self.iterate_list = []
        self.param_list = []
        self.iterate_to_id_map = {}
        self.id_to_iterate_map = {}
        self.init_iter_range_map = {}
        self.sample_iter_bound_map = {}

        self.range_marker = 0

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
        # print(self.range_marker)
        # print(self.init_iter_range_map)

    def create_sample_iter_bound_maps(self):
        print(self.num_samples)
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
        print(self.problem_dim)

    def preprocess_constraints(self):
        # after this method, map the sets to their constraint indices
        # also, precompute b_lower and b_upper for AX
        pass

    def solve(self, **kwargs):
        print('solving')

    def canonicalize(self, **kwargs):
        print('canonicalizing')
        self.convert_hl_to_basic_steps()
        self.create_iterate_id_maps()
        self.create_init_iterate_range_maps()
        self.create_sample_iter_bound_maps()
        self.set_problem_dim()

        self.preprocess_constraints()
