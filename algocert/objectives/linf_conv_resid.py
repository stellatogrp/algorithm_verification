from algocert.variables.iterate import Iterate


class LInfConvResid(object):
    """ Docstring for LInfConvResid. """

    def __init__(self, x: Iterate, M=100):
        self.x = x
        self.M = M

    def __str__(self):
        return f'OBJ: LINF_CONV_RESID({self.x.name})'

    def get_iterate(self):
        return self.x
