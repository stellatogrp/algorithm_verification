from algocert.variables.variable import Variable


class Iterate(Variable):

    """Docstring for Iterate. """

    def __init__(self, n, name='-'):
        super().__init__(n, name=name)
        self.depend_on_iter = True

    def __repr__(self):
        return f'ITERATE({self.name})'
