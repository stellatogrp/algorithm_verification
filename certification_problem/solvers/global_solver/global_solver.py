from certification_problem.solvers.solver import Solver
from certification_problem.solvers.global_solver.global_handler import GlobalHandler


class GlobalSolver(Solver):

    """Docstring for GlobalSolver. """

    def __init__(self, CP):
        """TODO: to be defined. """
        CP.print_cp()
        self.CP = CP
        self.handler = None

    def solve(self):
        # Create and solve with Gurobi
        self.handler.solve()

    def canonicalize(self):
        # Iterate through steps and canonicalize them
        handler = GlobalHandler(self.CP)
        self.handler = handler
        handler.canonicalize()
