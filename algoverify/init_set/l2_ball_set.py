from algoverify.init_set.init_set import InitSet


class L2BallSet(InitSet):

    def __init__(self, x, c, r=1, canon_iter=None):
        super().__init__(x, canon_iter=canon_iter)
        self.c = c
        self.r = r

    def __str__(self):
        to_string = f'SET({self.x.name}): l2 ball of radius {self.r} centered at {self.c}'
        return to_string

    def sample_point(self):
        return self.c
