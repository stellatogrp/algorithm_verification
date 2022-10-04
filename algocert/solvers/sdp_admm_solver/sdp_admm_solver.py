from algocert.solvers.sdp_admm_solver.sdp_admm_handler import SDPADMMHandler
from algocert.solvers.solver import Solver


class SDPADMMSolver(Solver):

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
        handler = SDPADMMHandler(self.CP, **kwargs)
        self.handler = handler
        handler.canonicalize()
