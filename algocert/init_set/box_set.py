
from algocert.init_set.init_set import InitSet


class BoxSet(InitSet):

    def __init__(self, x, l, u):
        super().__init__(x)
        self.l = l
        self.u = u

    def __str__(self):
        to_string = f'SET({self.x.name}): box with l = {self.l.reshape(-1,)} and u = {self.u.reshape(-1,)}'
        return to_string

    def __repr__(self):
        to_string = f'BOXSET({self.x.name})'
        return to_string

    def sample_point(self):
        # n = self.x.get_dim()
        # sample = np.random.uniform(size=(n, 1))
        # # print(sample)
        # return np.multiply((self.u - self.l), sample) + self.l
        return (self.u - self.l) / 2
