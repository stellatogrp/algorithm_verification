from algocert.solvers.sdp_cgal_solver.sdp_cgal_handler import SDPCGALHandler
from algocert.solvers.solver import Solver


class SDPCGALSolver(Solver):

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
        handler = SDPCGALHandler(self.CP, **kwargs)
        self.handler = handler
        handler.canonicalize()
