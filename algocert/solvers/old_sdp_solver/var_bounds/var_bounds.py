import cvxpy as cp


class CPVarAndBounds(object):

    def __init__(self, dims, l=None, u=None):
        self.dims = dims
        self.cp_var = cp.Variable(dims)
        self.l = l
        self.u = u

    def get_cp_var(self):
        return self.cp_var

    def get_lower_bound(self):
        return self.l

    def set_lower_bound(self, new_l):
        self.l = new_l

    def get_upper_bound(self):
        return self.u

    def set_upper_bound(self, new_u):
        self.u = new_u
