from algocert.init_set.init_set import InitSet


class CenteredL2BallSet(InitSet):

    def __init__(self, x, r=1):
        super().__init__(x)
        self.r = r

    def __str__(self):
        to_string = f'SET({self.x.name}): l2 ball of radius {self.r} centered at zero'
        return to_string
