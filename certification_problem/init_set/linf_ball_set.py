import numpy as np
from certification_problem.init_set.box_set import BoxSet


class LInfBallSet(BoxSet):

    def __init__(self, x, c, r):
        n = x.get_dim()
        l = c - r * np.ones((n, 1))
        u = c + r * np.ones((n, 1))
        super().__init__(x, l, u)
        self.c = c
        self.r = r

    def __str__(self):
        to_string = f'SET({self.x.name}): linf ball with r = {self.r} and c = {self.c}'
        return to_string
