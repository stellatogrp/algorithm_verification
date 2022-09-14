from algocert.variables.iterate import Iterate
from algocert.basic_algorithm_steps.step import Step


class MinWithVecStep(Step):

    """Docstring for MinWithVecStep. """

    def __init__(self, y: Iterate, x: Iterate, u=None):
        """Step representing y = min(x, u) elementwise
        """
        self.x = x
        self.y = y
        self.u = u
        self._test_dims()

    def _test_dims(self):
        if self.x.dim != self.y.dim:
            raise AssertionError('iterate dimensions for nonneg proj do not match')

    def __str__(self):
        return f'{self.y.name} = MIN_WITH_VEC({self.x.name})'

    def get_output_var(self):
        return self.y

    def get_input_var(self):
        return self.x

    def get_upper_bound_vec(self):
        return self.u

    #  def apply(self, x):
    #      return intermediate
