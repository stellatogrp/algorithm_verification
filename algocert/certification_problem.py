import algocert.settings as s
from algocert.solvers.global_solver.global_solver import GlobalSolver
from algocert.solvers.sdp_solver.sdp_solver import SDPSolver


class CertificationProblem(object):

    """Docstring for CertificationProblem. """

    def __init__(self, N, init_sets, parameter_sets, objective, algorithm,
                 qp_problem_data=None, add_RLT_constraints=False):
        self.N = N
        self.init_sets = init_sets
        self.parameter_sets = parameter_sets
        self.algorithm = algorithm
        self.objective = objective
        if qp_problem_data is not None:
            self.qp_problem_data = qp_problem_data
        else:
            self.qp_problem_data = {}
        self.add_RLT_constraints = add_RLT_constraints

    def solve(self, solver_type=s.DEFAULT, **kwargs):
        # Define and solve the problem
        if solver_type == s.SDP:
            solver = SDPSolver(self)
            solver.canonicalize(**kwargs)
            # TODO break this out and add a way to specify the variable
            # solver.handler.add_convexity_constraints(self.qp_problem_data['A'])
            res = solver.solve(**kwargs)
        if solver_type == s.GLOBAL:
            solver = GlobalSolver(self)
            solver.canonicalize(**kwargs)
            res = solver.solve(**kwargs)
        self.solver = solver
        return res

    def print_cp(self):
        print(f'{self.N} steps of algorithm')

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
