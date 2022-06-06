from init_set import InitSet


class BoxSet(InitSet):

    def __init__(self, l, u):
        super().__init__()
        self.l = l
        self.u = u

    def __str__(self):
        to_string = f'SET: box with l = {self.l} and u = {self.u}'
        return to_string
