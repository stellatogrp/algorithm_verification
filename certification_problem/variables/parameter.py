from certification_problem.variables.variable import Variable


class Parameter(Variable):

    """Docstring for Parameter. """

    def __init__(self, n, name):
        super().__init__(n, name)
        self.depend_on_iter = False
