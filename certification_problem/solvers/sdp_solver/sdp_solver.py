from certification_problem.solvers.solver import Solver
from certification_problem.solvers.sdp_solver.sdp_handler import SDPHandler


class SDPSolver(Solver):

    """Docstring for SDPSolver. """

    def __init__(self, CP):
        """TODO: to be defined. """
        CP.print_cp()
        self.CP = CP
        self.handler = None

    def solve(self):
        # Create SDP relaxation and solve
        self.handler.solve()

    def canonicalize(self):
        # Iterate through steps and canonicalize them
        handler = SDPHandler(self.CP)
        self.handler = handler
        handler.canonicalize()
