import time

from algoverify.solvers.sdp_custom_solver.sdp_custom_handler import SDPCustomHandler
from algoverify.solvers.solver import Solver


class SDPCustomSolver(Solver):

    """Docstring for SDPSolver. """

    def __init__(self, CP):
        """TODO: to be defined. """
        CP.print_cp()
        self.CP = CP
        self.handler = None

    def solve(self, **kwargs):
        # Create SDP relaxation and solve
        res = self.handler.solve()
        res['sdp_canontime'] = self.canon_time
        return res

    def canonicalize(self, **kwargs):
        # Iterate through steps and canonicalize them
        handler = SDPCustomHandler(self.CP, **kwargs)
        self.handler = handler
        start = time.time()
        handler.canonicalize()
        end = time.time()
        self.canon_time = end - start
