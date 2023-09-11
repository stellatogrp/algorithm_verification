import numpy as np


class RangeHandler2D(object):

    def __init__(self, ranges1, ranges2):
        if not isinstance(ranges1, list):
            self.ranges1 = [ranges1]
        else:
            self.ranges1 = ranges1

        if not isinstance(ranges2, list):
            self.ranges2 = [ranges2]
        else:
            self.ranges2 = ranges2

        self.process_dims()

    def process_dims(self):
        ranges1_dim = 0
        ranges2_dim = 0
        for r in self.ranges1:
            ranges1_dim += (r[1] - r[0])
        for r in self.ranges2:
            ranges2_dim += (r[1] - r[0])
        self.ranges1_dim = ranges1_dim
        self.ranges2_dim = ranges2_dim

    def index_matrix(self):
        row_indices = []
        col_indices = []
        for r in self.ranges1:
            row_indices += list(range(r[0], r[1]))
        for r in self.ranges2:
            col_indices += list(range(r[0], r[1]))
        return np.ix_(row_indices, col_indices)


class RangeHandler1D(RangeHandler2D):

    def __init__(self, ranges):
        super().__init__(ranges, (-1, 0))
