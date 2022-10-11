from algocert.variables.iterate import Iterate


class LinCombSquaredNorm(object):
    """ Docstring for LinCombSquaredNorm. """

    def __init__(self, A_stack, x_stack: [Iterate]):
        self.x_stack = x_stack
        self.A_stack = A_stack

    def get_iterate_stack(self):
        return self.x_stack

    def get_matrix_stack(self):
        return self.A_stack

    def __str__(self):
        return 'OBJ: LINCOMB_SQUARED_NORM()'

    def get_iterate(self):
        return self.x
