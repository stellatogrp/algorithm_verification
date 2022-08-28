from certification_problem.variables.iterate import Iterate
from certification_problem.basic_algorithm_steps.step import Step


class HighLevelLinearStep(Step):

    """Docstring for HighLevelLinearStep. """

    def __init__(self, y: Iterate, u: [Iterate], D=None, A=None, b=None):
        """Step representing Dy = A[u] + b

        Args:
            A (TODO): TODO
        """
        self.A = A
        self.D = D
        self.b = b
        self.u = u
        self.y = y
        self._test_dims()

    def _test_dims(self):
        # should be D: (n, k), y: (k, 1), A: (n, m), u: (m, 1), b: (n, 1)
        # TODO retool this to use Nones for D and b
        u_dim = 0
        for x in self.u:
            u_dim += x.get_dim()
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

    def get_output_var(self):
        return self.y

    def get_input_var(self):
        return self.u

    def get_rhs_matrix(self):
        return self.A

    def get_lhs_matrix(self):
        return self.D

    def get_rhs_const_vec(self):
        return self.b

    #  def apply(self, x):
    #      return intermediate
