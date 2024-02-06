import numpy as np
import scipy.sparse as spa

from algocert.basic_algorithm_steps.step import Step
from algocert.variables.iterate import Iterate


class LinearMaxProjStep(Step):

    """Docstring for LinearStep. """

    def __init__(self, y: Iterate, u: [Iterate], A=None, b=None, l=None, proj_ranges=None, start_canon=None):
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
        # self.b = b
        self.u = u
        self.y = y

        if l is not None:
            self.l = l
        else:
            n = y.get_dim()
            self.l = np.zeros((n, 1))

        self.is_linstep = False
        self.A_blocks = []
        self.A_boundaries = []
        self._test_dims()
        # self._split_A()

        if proj_ranges is None:
            self.proj_indices = list(range(0, y.get_dim()))
            self.nonproj_indices = []
        else:
            if isinstance(proj_ranges, list):
                self.proj_ranges = proj_ranges
            else:
                self.proj_ranges = [proj_ranges]
            self._compute_indices()

    def _test_dims(self):
        # should be D: (n, k), y: (k, 1), A: (n, m), u: (m, 1), b: (n, 1)
        # TODO retool this to use Nones for D and b
        u_dim = 0
        for x in self.u:
            u_dim += x.get_dim()
        self.u_dim = u_dim
        A_dim = self.A.shape
        y_dim = self.y.get_dim()
        b_dim = self.b.shape
        if A_dim[1] != u_dim:
            print(A_dim, u_dim)
            raise AssertionError('RHS A and u dimensions do not match')
        if A_dim[0] != b_dim[0]:
            print(A_dim, b_dim)
            raise AssertionError('RHS A and b dimensions do not match')
        if y_dim != A_dim[0]:
            raise AssertionError('LHS and RHS vector dimensions do not match')

    def _compute_indices(self):
        all_indices = set(range(self.y.dim))
        # real_indices = set()
        for range_bounds in self.proj_ranges:
            lo = range_bounds[0]
            hi = range_bounds[1]
            curr_range = set(range(lo, hi))
            all_indices = all_indices - curr_range
        nonproj_indices = all_indices.copy()
        proj_indices = set(range(self.y.dim)) - nonproj_indices
        self.proj_indices = list(proj_indices)
        self.nonproj_indices = list(nonproj_indices)

    def __str__(self):
        # return f'{self.y.name} = LINSTEP({self.x.name}) with matrix A = {self.A}'
        u_name = ''
        for x in self.u:
            u_name += x.get_name() + ', '
        return f'{self.y.name} = LINMAXPROJSTEP({u_name})'

    def __repr__(self):
        return self.__str__()

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

    def get_rhs_const_vec(self):
        return self.b

    def get_lower_bound_vec(self):
        return self.l

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

    def get_matrix_data(self, k):
        if self.A_list is not None:
            A = self.A_list[k-1]
        else:
            A = self.A

        if self.b_list is not None:
            b = self.b_list[k-1]
        else:
            b = self.b

        return dict(A=spa.csc_matrix(A), b=b)
