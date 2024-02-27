import numpy as np

from algoverify.init_set.l2_ball_set import L2BallSet


class ZeroSet(L2BallSet):

    def __init__(self, x):
        z = np.zeros((x.get_dim(), 1))
        super().__init__(x, z, 0)

    def __str__(self):
        to_string = f'ZEROSET({self.x.name})'
        return to_string
