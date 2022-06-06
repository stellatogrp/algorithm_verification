import certification_problem.settings as s


class CertificationProblem(object):

    """Docstring for CertificationProblem. """

    def __init__(self, N, init_set, parameter_set, objective, algorithm):
        # number of steps
        self.N = N
        # initial iterates set
        self.init_set = init_set
        # parameter set
        self.parameter_set = parameter_set
        # objective
        self.objective = objective
        # algorithm
        self.algorithm = algorithm

    def solve(solver=s.DEFAULT):
        # Define and solve the problem
        pass

