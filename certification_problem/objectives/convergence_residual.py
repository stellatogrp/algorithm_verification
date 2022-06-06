from certification_problem.variables.iterate import Iterate


class ConvergenceResidual(object):
    """ Docstring for ConvergenceResidual. """

    def __init__(self, x: Iterate):
        self.x = x

    def __str__(self):
        return f'objective: CONVERGENCE_RESIDUAL({self.x.name})'
