from algocert.solvers.sdp_scgal_solver.sdp_scgal_handler import SDPSCGALHandler
from algocert.solvers.solver import Solver


class SDPSCGALSolver(Solver):

    def __init__(self, CP):
        CP.print_cp()
        self.CP = CP
        self.handler = None

    def solve(self, **kwargs):
        # Create SDP relaxation and solve
        res = self.handler.solve(**kwargs)
        return res

    def canonicalize(self, **kwargs):
        # Iterate through steps and canonicalize them
        handler = SDPSCGALHandler(self.CP, **kwargs)
        self.handler = handler
        handler.canonicalize()
