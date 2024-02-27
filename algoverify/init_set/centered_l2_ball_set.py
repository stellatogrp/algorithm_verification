import numpy as np

from algoverify.init_set.init_set import InitSet


class CenteredL2BallSet(InitSet):

    def __init__(self, x, r=1):
        super().__init__(x)
        self.r = r

    def __str__(self):
        to_string = f'SET({self.x.name}): l2 ball of radius {self.r} centered at zero'
        return to_string

    def sample_point(self):
        '''
        https://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability/87238#87238
        '''
        n = self.x.get_dim()
        sample = np.random.randn(n, 1)
        surface_point = sample / np.linalg.norm(sample)
        U = np.random.uniform()
        return self.r * np.power(U, 1/n) * surface_point
