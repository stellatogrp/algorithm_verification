from algocert.init_set.init_set import InitSet


class ConstSet(InitSet):

    def __init__(self, x, val):
        super().__init__(x)
        self.val = val

    def get_val(self):
        return self.val

    def __str__(self):
        to_string = f'SET({self.x.name}): const = {self.val.reshape(-1,)}'
        return to_string
