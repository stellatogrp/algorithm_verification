from init_set import InitSet


class L2BallSet(InitSet):

    def __init__(self, c, r):
        super().__init__()
        self.c = c
        self.r = r

    def __str__(self):
        to_string = self.print_prefix + f'l2 ball of radius {self.r} centered at {self.c}'
        return to_string
