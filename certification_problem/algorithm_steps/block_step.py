from certification_problem.variables.variable import Variable
from certification_problem.variables.iterate import Iterate
from certification_problem.variables.parameter import Parameter


class BlockStep(object):

    """Docstring for LinearStep. """

    def __init__(self, u: Iterate, list_x: [Variable]):
        """ Step representing u = [list_x], i.e. make a block vector where u represents
                the stack of variables from list_x,

        Args:
            u (TODO): TODO
            list_x (TODO): TODO
        """
        self.u = u
        self.list_x = list_x
        self._test_dims()

    def _test_dims(self):
        block_dim = 0
        for x in self.list_x:
            block_dim += x.dim
        if self.u.dim != block_dim:
            raise AssertionError('dimensions of block do not match')

    def __str__(self):
        list_str = '['
        for x in self.list_x:
            list_str += f'{x.name}, '
        list_str += ']'
        return f'{self.u.name} = BLOCKSTEP({list_str})'


x1 = Iterate(5, 'x1')
x2 = Iterate(3, 'x2')
x3 = Iterate(4, 'x3')
b = Parameter(1, 'b')
y = Iterate(13, 'y')
step = BlockStep(y, [x1, x2, x3, b])
print(step)
