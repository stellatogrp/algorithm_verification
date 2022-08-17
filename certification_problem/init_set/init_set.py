from certification_problem.variables.variable import Variable


class InitSet(object):

    """Doc string"""

    def __init__(self, x: Variable):
        """TODO: to be defined."""
        self.x = x

    def get_iterate(self):
        return self.x

    def sdr_canonicalizer(self):
        raise NotImplementedError
