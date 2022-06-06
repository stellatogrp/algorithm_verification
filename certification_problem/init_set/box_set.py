from certification_problem.init_set.init_set import InitSet


class BoxSet(InitSet):

    def __init__(self, x, l, u):
        super().__init__(x)
        self.l = l
        self.u = u

    def __str__(self):
        to_string = f'SET({self.x.name}): box with l = {self.l} and u = {self.u}'
        return to_string
