from algoverify.basic_algorithm_steps.step import Step
from algoverify.variables.iterate import Iterate


class NonNegLinStep(Step):

    """Docstring for NonNegLinStep. """

    def __init__(self, y: Iterate, u: [Iterate], C=None, b=None):
        """Step representing y = (C[u] + b)_+
        """
        self.y = y
        self.C = C
        if isinstance(u, list):
            self.u = u
        else:
            self.u = [u]
        self.b = b.reshape(-1, 1)
        self._test_dims()
        self._split_C()

    def _test_dims(self):
        # should be y: (n, 1), A: (n, m), u: (m, 1), b: (n, 1)
        u_dim = 0
        for x in self.u:
            u_dim += x.get_dim()
        self.u_dim = u_dim
        C_dim = self.C.shape
        y_dim = self.y.get_dim()
        b_dim = self.b.shape[0]
        if C_dim[1] != u_dim:
            print(C_dim, u_dim)
            raise AssertionError('RHS C and u dimensions do not match')
        if C_dim[0] != b_dim:
            print(C_dim, b_dim)
            raise AssertionError('RHS C and b dimensions do not match')
        if C_dim[0] != y_dim:
            print(C_dim, y_dim)
            raise AssertionError('LHS and RHS vector dimensions do not match')

    def _split_C(self):
        C_blocks = []
        C = self.C
        # left = 0
        # right = 0
        # boundaries = []
        # for x in u:
        #     n = x.get_dim()
        #     right = left + n
        #     # print(left, right)
        #     # print(A.tocsc()[:, left: right].shape)
        #     boundaries.append((left, right))
        #     left = right
        left = 0
        right = 0
        for x in self.u:
            n = x.get_dim()
            right = left + n
            C_blocks.append(C.tocsc()[:, left: right])
            left = right
        self.C_blocks = C_blocks

    def __str__(self):
        # return f'{self.y.name} = LINSTEP({self.x.name}) with matrix A = {self.A}'
        u_name = ''
        for x in self.u:
            u_name += x.get_name() + ', '
        return f'{self.y.name} = NONNEGLINSTEP({u_name})'

    def __repr__(self):
        return self.__str__()

    def get_output_var(self):
        return self.y

    def get_input_var(self):
        return self.u

    def get_rhs_matrix(self):
        return self.C

    def get_rhs_const_vec(self):
        return self.b
