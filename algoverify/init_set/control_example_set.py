from algoverify.init_set.init_set import InitSet


class ControlExampleSet(InitSet):

    def __init__(self, x, l, u):
        super().__init__(x)
        self.l = l
        self.u = u

    def get_nonzero(self):
        return self.x.get_dim()

    def __str__(self):
        to_string = f'SET({self.x.name}): ControlSet = '
        return to_string
