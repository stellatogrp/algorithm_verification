from algocert.variables.variable import Variable


class InitSet(object):

    """Doc string"""

    def __init__(self, x: Variable, canon_iter=None):
        """TODO: to be defined."""
        self.x = x
        if canon_iter is None:
            self.canon_iter = set([0])
        else:
            self.canon_iter = set(canon_iter)

    def get_iterate(self):
        return self.x

    def sdr_canonicalizer(self):
        raise NotImplementedError

    def sample_point(self):
        raise NotImplementedError
