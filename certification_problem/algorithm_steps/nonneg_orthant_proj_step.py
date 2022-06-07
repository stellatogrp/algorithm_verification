from certification_problem.variables.iterate import Iterate


class NonNegProjStep(object):

    """Docstring for NonNegStep. """

    def __init__(self, y: Iterate, x: Iterate):
        """Step representing y = (x)_+
        """
        self.x = x
        self.y = y
        self._test_dims()

    def _test_dims(self):
        if self.x.dim != self.y.dim:
            raise AssertionError('iterate dimensions for nonneg proj do not match')

    def __str__(self):
        return f'{self.y.name} = NONNEG_PROJ({self.x.name})'

    def get_output_var(self):
        return self.y

    def get_input_var(self):
        return self.x

    #  def apply(self, x):
    #      return intermediate
