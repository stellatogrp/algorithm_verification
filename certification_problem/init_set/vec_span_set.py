from certification_problem.init_set.init_set import InitSet


class VecSpanSet(InitSet):

    def __init__(self, x, v, a, b):
        super().__init__(x)
        self.v = v
        self.a = a
        self.b = b

    def __str__(self):
        to_string = f'SET({self.x.name}): vector span set with {self.a} <= c <= {self.b} and span of {self.v}'
        return to_string
