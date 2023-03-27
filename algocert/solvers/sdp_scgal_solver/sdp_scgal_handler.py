import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_scgal_solver import (OBJ_CANON_METHODS,
                                               SET_PREPROCESS_METHODS,
                                               SET_PRIMITIVE_2_METHODS,
                                               STEP_PREPROCESS_METHODS,
                                               STEP_PRIMITIVE_2_METHODS)


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

        self.constr_counter = 1
        self.b_lower = [[1]]
        self.b_upper = [[1]]

        self.init_set_constraint_bounds = {}
        self.sample_param_constraint_bounds = {}
        self.sample_step_constraint_bounds = {}

        self.primitive1 = None
        self.primitive2 = None
        self.primitive3 = None

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
        print('num samples:', self.num_samples)
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

        print('sample iter bound map:', self.sample_iter_bound_map)

    def set_problem_dim(self):
        self.problem_dim = self.range_marker + 1
        print('problem dim:', self.problem_dim)

    def update_constr_counter(self, num_constr):
        start = self.constr_counter
        end = self.constr_counter + num_constr
        self.constr_counter = end
        return start, end

    def preprocess_constraints(self):
        # after this method, map the sets to their constraint indices
        # also, precompute b_lower and b_upper for AX
        self.preprocess_initial_sets()
        self.preprocess_parameter_sets()
        self.preprocess_steps()
        # print(np.vstack(self.b_lower), np.vstack(self.b_upper), self.constr_counter)

        # also add bound propagation here

    def preprocess_initial_sets(self):
        for init_set in self.CP.get_init_sets():
            canon_method = SET_PREPROCESS_METHODS[type(init_set)]
            num_constr, b_l, b_u = canon_method(init_set)
            self.b_lower.append(b_l)
            self.b_upper.append(b_u)
            start, end = self.update_constr_counter(num_constr)
            self.init_set_constraint_bounds[init_set] = (start, end)
        print('init constr bounds:', self.init_set_constraint_bounds)

    def preprocess_parameter_sets(self):
        # note this differs from the init_sets because we have to do this per sample
        for i in range(self.num_samples):
            sample_dict = {}
            for param_set in self.CP.get_parameter_sets():
                canon_method = SET_PREPROCESS_METHODS[type(param_set)]
                num_constr, b_l, b_u = canon_method(param_set)
                self.b_lower.append(b_l)
                self.b_upper.append(b_u)
                start, end = self.update_constr_counter(num_constr)
                sample_dict[param_set] = (start, end)
            self.sample_param_constraint_bounds[i] = sample_dict
        # print(self.constr_counter)
        print('sample param constraint bounds:', self.sample_param_constraint_bounds)

    def preprocess_steps(self):
        for i in range(self.num_samples):
            curr_sample_bound_map = self.sample_iter_bound_map[i]
            print('sample:', i, 'bounds:', curr_sample_bound_map)
            steps = self.alg_steps
            sample_dict = {}
            for step in steps:
                canon_method = STEP_PREPROCESS_METHODS[type(step)]
                step_dict = {}
                for k in range(1, self.K + 1):
                    num_constr, b_l, b_u = canon_method(step)
                    self.b_lower.append(b_l)
                    self.b_upper.append(b_u)
                    start, end = self.update_constr_counter(num_constr)
                    step_dict[k] = (start, end)
                sample_dict[step] = step_dict
            self.sample_step_constraint_bounds[i] = sample_dict
        print('sample step contr bounds:', self.sample_step_constraint_bounds)
        print('number of constraints:', self.constr_counter)
        # print(np.vstack(self.b_lower), np.vstack(self.b_upper))
        # print(len(np.vstack(self.b_lower)), len(np.vstack(self.b_upper)))

    def create_primitives(self):
        self.primitive1 = self.create_primitive_1()
        self.primitive2 = self.create_primitive_2()
        # test = self.primitive2(np.zeros(self.problem_dim), np.zeros(self.constr_counter))
        test = self.primitive2(np.ones(self.problem_dim), np.ones(self.constr_counter))
        print(test)

    def create_primitive_1(self):
        """
            Primitive 1 is u -> Cu, u in R^n
            where C is for the objective, i.e. tr(CX)

            For primitive 1, since its just one matrix, we create and store
                the entire matrix C
        """
        if type(self.CP.objective) != list:
            obj_list = [self.CP.objective]
        else:
            obj_list = self.CP.objective
        C = 0
        for obj in obj_list:
            obj_canon = OBJ_CANON_METHODS[type(obj)]
            C += obj_canon(obj, self)
        C = spa.csc_matrix(C)

        def p1(u):
            return C @ u

        return p1

    def create_primitive_2(self):
        """
            Primitive 2 is (u, z) -> (A^\star z) u
                where z in R^d (d is number of constraints)
            and (A^\star z) = \sum_{i=1}^d z_i A_i

            Idea: with all the auxiliary set up, can extract the z_i corresponding
                to the relevant constraint indices, then canonicalize based on the
                set or step

                It is crucial that we following the exact same canonicalization order
                lower right 1 constraint -> init sets -> (per sample) param sets ->
                    (per sample) (per step) (1 through K) steps
        """
        def p2(u, z):
            # out = np.zeros((self.problem_dim, 1))
            out = np.zeros(self.problem_dim)
            # first, the lower right constraint, i.e. X[-1, -1] = 1
            out[-1] = z[0] * u[-1]

            # first, initial sets:
            for init_set in self.CP.get_init_sets():
                canon_method = SET_PRIMITIVE_2_METHODS[type(init_set)]
                x_indices = self.init_iter_range_map[init_set.get_iterate()]
                z_indices = self.init_set_constraint_bounds[init_set]
                print(z_indices)
                z_vals = z[z_indices[0]: z_indices[1]]
                out += canon_method(u, init_set, x_indices, z_vals, self)

            # then, parameter sets:
            for i in range(self.num_samples):
                # self.sample_param_constraint_bounds = {}
                sample_param_var_bounds = self.sample_iter_bound_map[i]
                sample_param_constr_bounds = self.sample_param_constraint_bounds[i]
                print(sample_param_var_bounds, sample_param_constr_bounds)
                for param_set in self.CP.get_parameter_sets():
                    canon_method = SET_PRIMITIVE_2_METHODS[type(param_set)]
                    x_indices = sample_param_var_bounds[param_set.get_iterate()]
                    z_indices = sample_param_constr_bounds[param_set]
                    print(z_indices)
                    z_vals = z[z_indices[0]: z_indices[1]]
                    out += canon_method(u, param_set, x_indices, z_vals, self)

            # lastly, the steps:
            for i in range(self.num_samples):
                steps = self.alg_steps
                for step in steps:
                    for k in range(1, self.K + 1):
                        canon_method = STEP_PRIMITIVE_2_METHODS[type(step)]

            return out

        return p2

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
        self.create_primitives()
