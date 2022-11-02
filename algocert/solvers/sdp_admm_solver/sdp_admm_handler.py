# from algocert.solvers.sdp_admm_solver import HL_TO_BASIC_STEP_METHODS


class SDPADMMHandler(object):

    def __init__(self, CP, **kwargs):
        self.CP = CP
        self.N = CP.N
        self.alg_steps = CP.get_algorithm_steps()
        self.iterate_list = []
        self.param_list = []

    def convert_hl_to_basic_steps(self):
        pass

    def canonicalize(self):
        self.convert_hl_to_basic_steps()

    def solve(self):
        return None
