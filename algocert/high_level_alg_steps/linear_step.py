import numpy as np

from algocert.basic_algorithm_steps.step import Step
from algocert.variables.iterate import Iterate


class LinearStep(Step):

    """Docstring for LinearStep. """

    def __init__(self, y: Iterate, u: [Iterate], D=None, A=None, b=None, Dinv=None):
        """Step representing Dy = A[u] + b

        Args:
            A (TODO): TODO
        """
        super().__init__()
        self.A = A
        self.D = D
        self.b = b
        self.u = u
        self.y = y
        self.Dinv = Dinv
        self.is_linstep = True
        self._test_dims()

    def _test_dims(self):
        # should be D: (n, k), y: (k, 1), A: (n, m), u: (m, 1), b: (n, 1)
        # TODO retool this to use Nones for D and b
        u_dim = 0
        for x in self.u:
            u_dim += x.get_dim()
        self.u_dim = u_dim
        D_dim = self.D.shape
        A_dim = self.A.shape
        y_dim = self.D.shape
        b_dim = self.b.shape
        if D_dim[1] != y_dim[0]:
            raise AssertionError('LHS D and y dimensions do not match')
        if A_dim[1] != u_dim:
            print(A_dim, u_dim)
            raise AssertionError('RHS A and u dimensions do not match')
        if A_dim[0] != b_dim[0]:
            print(A_dim, b_dim)
            raise AssertionError('RHS A and b dimensions do not match')
        if D_dim[0] != A_dim[0]:
            print(D_dim, A_dim)
            raise AssertionError('LHS and RHS vector dimensions do not match')

    def __str__(self):
        # return f'{self.y.name} = LINSTEP({self.x.name}) with matrix A = {self.A}'
        u_name = ''
        for x in self.u:
            u_name += x.get_name() + ', '
        return f'{self.y.name} = LINSTEP({u_name})'

    # def _split_A(self):
    #     A_blocks = []
    #     C = self.CGALMaxCutTester
    #     left = 0
    #     right = 0
    #     for x in self.u:
    #         n = x.get_dim()
    #         right = left + n
    #         C_blocks.append(C.tocsc()[:, left: right])
    #         left = right
    #     self.C_blocks = C_blocks

    def get_output_var(self):
        return self.y

    def get_input_var(self):
        return self.u

    def get_rhs_matrix(self):
        return self.A

    def get_lhs_matrix(self):
        return self.D

    def get_lhs_matrix_inv(self):
        return self.Dinv

    def get_rhs_const_vec(self):
        return self.b

    #  def apply(self, x):
    #      return intermediate

    def map_overall_dim_to_x(self, i):
        assert 0 <= i and i < self.u_dim
        curr_dim = 0
        for x in self.u:
            x_dim = x.get_dim()
            if curr_dim <= i and i < curr_dim + x_dim:
                return x
            curr_dim += x_dim

        #     for x in u:
        # if not x.is_param:
        #     if iter_to_id_map[y] <= iter_to_id_map[x]:
        #         iter_to_k_map[x] = k-1
        #     else:
        #         iter_to_k_map[x] = k

    def apply(self, k, iter_to_id_map, ranges, out):
        # TODO, this should really be handled in the cgal handler and not here
        y = self.y
        u_vec = []
        for x in self.u:
            if not x.is_param:
                if iter_to_id_map[y] <= iter_to_id_map[x]:
                    x_range = ranges[k-1][x]
                else:
                    x_range = ranges[k][x]
            else:
                x_range = ranges[x]
            u_vec.append(out[x_range[0]: x_range[1]])
        # print(np.vstack(u_vec))
        return self.A @ np.vstack(u_vec)
