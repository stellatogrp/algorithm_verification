import numpy as np
import scipy as sp

from algoverify.basic_algorithm_steps.step import Step
from algoverify.variables.iterate import Iterate


class LinearStep(Step):

    """Docstring for LinearStep. """

    def __init__(self, y: Iterate, u: [Iterate], D=None, A=None, b=None, Dinv=None, start_canon=None):
        """Step representing Dy = A[u] + b

        Args:
            A (TODO): TODO
        """
        super().__init__(start_canon=start_canon)

        if isinstance(A, list):
            # TODO: fix this hack, currently use A for the first one as a proxy to check dims etc
            self.A_list = A
            self.A = A[0]
        else:
            self.A_list = None
            self.A = A

        if isinstance(b, list):
            self.b_list = b
            self.b = b[0]
        else:
            self.b_list = None
            self.b = b

        # self.A = A
        self.D = D
        # self.b = b
        self.u = u
        self.y = y
        self.Dinv = Dinv
        self.is_linstep = True
        self.A_blocks = []
        self.A_boundaries = []
        self._test_dims()
        # self._split_A()
        self.D_factor = sp.linalg.lu_factor(D.todense())

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

    def __repr__(self):
        return self.__str__()

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

    def get_input_var_dim(self):
        return self.u_dim

    def get_rhs_matrix(self):
        return self.A

    def get_rhs_matrix_blocks(self):
        return self.A_blocks

    def get_lhs_matrix(self):
        return self.D

    def get_lhs_matrix_inv(self):
        return self.Dinv

    def get_rhs_const_vec(self):
        return self.b

    #  def apply(self, x):
    #      return intermediate

    # def _split_A(self):
    #     self._split_A_boundaries()
    #     A = self.A
    #     for i, x in enumerate(self.u):
    #         (left, right) = self.A_boundaries[i]
    #         self.A_blocks.append(A.tocsc()[:, left: right])

    # def _split_A_boundaries(self):
    #     left = 0
    #     right = 0
    #     for x in self.u:
    #         n = x.get_dim()
    #         right = left + n
    #         self.A_boundaries.append((left, right))
    #         left = right

    def split_matrix_boundaries(self):
        left = 0
        right = 0
        A_bounds = []
        for x in self.u:
            n = x.get_dim()
            right = left + n
            A_bounds.append((left, right))
            left = right
        return A_bounds

    def split_matrix(self, A):
        left = 0
        right = 0
        A_list = []
        for x in self. u:
            n = x.get_dim()
            right = left + n
            A_list.append(A[:, left: right])
            left = right
        return A_list

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

    def solve_linear_system(self, b):
        '''
            solves D x = b, where b is a vector or matrix
        '''
        return sp.linalg.lu_solve(self.D_factor, b)

    def get_matrix_data(self, k):
        if self.A_list is not None:
            A = self.A_list[k-1]
        else:
            A = self.A

        if self.b_list is not None:
            b = self.b_list[k-1]
        else:
            b = self.b

        return dict(D=self.D, A=A, b=b)

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