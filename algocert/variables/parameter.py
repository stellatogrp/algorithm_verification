from algocert.variables.variable import Variable


class Parameter(Variable):

    """Docstring for Parameter. """

    def __init__(self, n, name='-'):
        super().__init__(n, name=name)
        self.depend_on_iter = False
        self.is_param = True

    def __repr__(self):
        return f'PARAMETER({self.name})'
