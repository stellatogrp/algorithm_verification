import certification_problem.settings as s


class CertificationProblem(object):

    """Docstring for CertificationProblem. """

    def __init__(self, N, init_sets, parameter_sets, objective, algorithm):
        # number of steps
        self.N = N
        # initial iterates set
        self.init_sets = init_sets
        # parameter set
        self.parameter_sets = parameter_sets
        # objective
        self.objective = objective
        # algorithm
        self.algorithm = algorithm

    def solve(solver=s.DEFAULT):
        # Define and solve the problem
        pass

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

        # print('----Objective----')
        # print(self.objective)
