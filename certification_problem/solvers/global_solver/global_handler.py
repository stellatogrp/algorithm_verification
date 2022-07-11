import gurobipy as gp

from gurobipy import GRB


class GlobalHandler(object):

    def __init__(self, CP):
        self.CP = CP

    def canonicalize(self):
        pass

    def solve(self):
        pass
