from init_set import InitSet


class L2BallSet(InitSet):

    def __init__(self, x, c, r):
        super().__init__(x)
        self.c = c
        self.r = r

    def __str__(self):
        to_string = f'SET({self.x.name}): l2 ball of radius {self.r} centered at {self.c}'
        return to_string
