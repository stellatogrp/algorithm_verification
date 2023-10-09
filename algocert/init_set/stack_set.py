from algocert.init_set.init_set import InitSet


class StackSet(InitSet):

    def __init__(self, x, stack):
        super().__init__(x)
        self.stack = stack

    def __str__(self):
        to_string = f'SET({self.x.name}): StackSet = '
        return to_string
