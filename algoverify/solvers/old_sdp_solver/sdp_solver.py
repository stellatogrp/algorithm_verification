from algoverify.solvers.sdp_solver.sdp_handler import SDPHandler
from algoverify.solvers.solver import Solver


class SDPSolver(Solver):

    """Docstring for SDPSolver. """

    def __init__(self, CP):
        """TODO: to be defined. """
        CP.print_cp()
        self.CP = CP
        self.handler = None

    def solve(self, **kwargs):
        # Create SDP relaxation and solve
        res = self.handler.solve(**kwargs)
        return res

    def canonicalize(self, **kwargs):
        # Iterate through steps and canonicalize them
        handler = SDPHandler(self.CP, **kwargs)
        self.handler = handler
        handler.canonicalize()
