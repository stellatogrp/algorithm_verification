class Variable(object):

    """Docstring for Variable."""

    def __init__(self, n, name='-'):
        self.dim = n
        self.name = name

    def get_name(self):
        return self.name

    def get_dim(self):
        return self.dim
