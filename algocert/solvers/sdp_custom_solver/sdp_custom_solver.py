from algocert.solvers.sdp_custom_solver.sdp_custom_handler import SDPCustomHandler
from algocert.solvers.solver import Solver


class SDPCustomSolver(Solver):

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
        handler = SDPCustomHandler(self.CP, **kwargs)
        self.handler = handler
        handler.canonicalize()
