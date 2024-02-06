from algoverify.variables.iterate import Iterate


class L1ConvResid(object):
    """ Docstring for L1ConvResid. """

    def __init__(self, x: Iterate):
        self.x = x

    def __str__(self):
        return f'OBJ: L1_CONVERGENCE_RESIDUAL({self.x.name})'

    def get_iterate(self):
        return self.x
