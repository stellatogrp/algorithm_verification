from algoverify.init_set.init_set import InitSet


class BoxStackSet(InitSet):

    def __init__(self, x, var_stack):
        super().__init__(x)
        self.var_stack = var_stack

    def __str__(self):
        to_string = f'SET({self.x.name}): BoxStackSet = '
        return to_string
