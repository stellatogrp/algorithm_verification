from algocert.basic_algorithm_steps.step import Step
from algocert.variables.iterate import Iterate


class MaxWithVecStep(Step):

    """Docstring for MaxWithVecStep. """

    def __init__(self, y: Iterate, x: Iterate, l=None):
        """Step representing y = max(x, l) elementwise
        """
        self.x = x
        self.y = y
        self.l = l
        self._test_dims()

    def _test_dims(self):
        if self.x.dim != self.y.dim:
            raise AssertionError('iterate dimensions for nonneg proj do not match')

    def __str__(self):
        return f'{self.y.name} = MAX_WITH_VEC({self.x.name})'

    def get_output_var(self):
        return self.y

    def get_input_var(self):
        return self.x

    def get_lower_bound_vec(self):
        return self.l

    #  def apply(self, x):
    #      return intermediate