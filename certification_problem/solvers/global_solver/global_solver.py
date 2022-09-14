from certification_problem.solvers.solver import Solver
from certification_problem.solvers.global_solver.global_handler import GlobalHandler


class GlobalSolver(Solver):

    """Docstring for GlobalSolver. """

    def __init__(self, CP):
        CP.print_cp()
        self.CP = CP
        self.handler = None

    def solve(self):
        # Create and solve with Gurobi
        if self.handler is None:
            raise AssertionError('Certification Problem has not been canonicalized yet.')
        res = self.handler.solve()
        return res

    def canonicalize(self, **kwargs):
        # Iterate through steps and canonicalize them
        handler = GlobalHandler(self.CP, **kwargs)
        self.handler = handler
        handler.canonicalize()
