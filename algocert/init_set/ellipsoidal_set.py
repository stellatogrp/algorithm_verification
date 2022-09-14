from algocert.init_set.init_set import InitSet


class EllipsoidalSet(InitSet):

    def __init__(self, x, Q, c):
        super().__init__(x)
        self.Q = Q
        self.c = c

    def __str__(self):
        to_string = f'SET({self.x.name}): ellipsoid centered at {self.c} with matrix Q'
        return to_string
