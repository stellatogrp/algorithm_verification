from algocert.init_set.init_set import InitSet


class VecSpanSet(InitSet):

    def __init__(self, x, v, a, b):
        super().__init__(x)
        self.v = v
        self.a = a
        self.b = b
        self.c_var = None
        self.csq_var = None

    def __str__(self):
        to_string = f'SET({self.x.name}): vector span set with {self.a} <= c <= {self.b} and span of {self.v}'
        return to_string

    def set_c_vars(self, c_var, csq_var):
        # this method gives us a way to store the cvxpy variables inside the set if we need it later
        self.c_var = c_var
        self.csq_var = csq_var
