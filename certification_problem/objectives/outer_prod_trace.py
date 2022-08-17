from certification_problem.variables.iterate import Iterate


class OuterProdTrace(object):
    """ Docstring for OuterProdTrace. """

    def __init__(self, x: Iterate):
        self.x = x

    def __str__(self):
        return f'OBJ: OUTER_PROD_TRACE({self.x.name})'

    def get_iterate(self):
        return self.x
