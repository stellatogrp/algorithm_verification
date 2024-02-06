from algoverify.variables.iterate import Iterate


class BlockConvergenceResidual(object):
    """ Docstring for ConvergenceResidual. """

    def __init__(self, s: [Iterate], A):
        self.s = s
        self.A = A

    def __str__(self):
        return f'OBJ: BLOCK_CONVERGENCE_RESIDUAL({self.s})'

    def get_iterate(self):
        return self.s

    def get_block_mat(self):
        return self.A

    def split_matrix_boundaries(self):
        left = 0
        right = 0
        A_bounds = []
        for x in self.s:
            n = x.get_dim()
            right = left + n
            A_bounds.append((left, right))
            left = right
        return A_bounds

    def split_matrix(self, A):
        left = 0
        right = 0
        A_list = []
        for x in self.s:
            n = x.get_dim()
            right = left + n
            A_list.append(A[:, left: right])
            left = right
        return A_list
