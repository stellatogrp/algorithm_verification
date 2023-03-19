import algocert.settings as s


class CertificationProblem(object):

    """Docstring for CertificationProblem. """
    def __init__(self, K, init_sets, parameter_sets, objective, algorithm,
                 num_samples=1):
        """
            K: the number of iterations
            init_sets: 
            parameter_sets: 
            objective: 
            algorithm: a list of steps

            i.e. something like
            steps = [step1, step2, step3, step4]
            zset = ConstSet(z, np.zeros((z_size, 1)))
            qset = BoxSet(q, lower, upper)
            self.obj = [ConvergenceResidual(z)]
            CP = CertificationProblem(N, [zset], [qset], obj, self.steps)
        """
        self.solver = None
        self.K = K
        self.num_samples = num_samples
        self.init_sets = init_sets
        self.parameter_sets = parameter_sets
        self.algorithm = algorithm
        self.objective = objective

    def solve(self, solver_type=s.DEFAULT, **kwargs):
        # Define and solve the problem
        if self.solver is not None:
            res = self.solver.solve(**kwargs)
        else:
            solver = s.solver_mapping[solver_type](self)
            self.solver = solver
            solver.canonicalize(**kwargs)
            res = solver.solve(**kwargs)

        return res

    def canonicalize(self, solver_type=s.DEFAULT, **kwargs):
        if self.solver is not None:
            self.solver.canonicalize(**kwargs)
        else:
            solver = s.solver_mapping[solver_type](self)
            self.solver = solver
            solver.canonicalize(**kwargs)
        return solver

    def print_cp(self):
        print(f'{self.K} steps of algorithm')

        print('----Initial set----')
        for init_set in self.init_sets:
            print(init_set)

        print('----Parameter set----')
        for param_set in self.parameter_sets:
            print(param_set)

        print('----Algorithm steps----')
        for step in self.algorithm:
            print(step)

        print('----Objective----')
        print(self.objective)

    def get_parameter_sets(self):
        return self.parameter_sets

    def get_init_sets(self):
        return self.init_sets

    def get_algorithm_steps(self):
        return self.algorithm

    def set_algorithm_steps(self, new_steps):
        self.algorithm = new_steps

    def get_param_map(self):
        return self.solver.handler.get_param_var_map()

    def get_iterate_map(self):
        return self.solver.handler.get_iterate_var_map()
