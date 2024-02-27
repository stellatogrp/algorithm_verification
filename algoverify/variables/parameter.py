from algoverify.variables.variable import Variable


class Parameter(Variable):
    """
    Parameter is a subclass of algoverify's Variable object
    """

    def __init__(self, n, name='-'):
        super().__init__(n, name=name)
        self.depend_on_iter = False
        self.is_param = True

    def __repr__(self):
        return f'PARAMETER({self.name})'
