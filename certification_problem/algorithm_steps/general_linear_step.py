from certification_problem.variables.iterate import Iterate
from certification_problem.algorithm_steps.step import Step


class LinearStep(Step):

    """Docstring for LinearStep. """

    def __init__(self, D, y: Iterate, A, x: Iterate, b):
        """Step representing Dy = Ax + b

        Args:
            A (TODO): TODO
        """
        self.A = A
        self.D = D
        self.b = b
        self.x = x
        self.y = y
        self._test_dims()

    def _test_dims(self):
        (m, n) = self.A.shape
        if self.x.dim != n:
            raise AssertionError('iterate dimension does not match RHS of matrix')
        if self.y.dim != m:
            raise AssertionError('iterate dimension does not match LHS of matrix')

    def __str__(self):
        # return f'{self.y.name} = LINSTEP({self.x.name}) with matrix A = {self.A}'
        return f'{self.y.name} = GENLINSTEP({self.x.name})'

    def get_output_var(self):
        return self.y

    def get_rhs_var(self):
        return self.x

    def get_rhs_matrix(self):
        return self.A

    def get_lhs_matrix(self):
        return self.D

    def get_rhs_const_vec(self):
        return self.b

    #  def apply(self, x):
    #      return intermediate
