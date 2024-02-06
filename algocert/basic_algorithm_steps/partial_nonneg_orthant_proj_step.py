# import numpy as np

from algocert.basic_algorithm_steps.step import Step
from algocert.variables.iterate import Iterate


class PartialNonNegProjStep(Step):

    def __init__(self, y: Iterate, x: Iterate, nonneg_ranges):
        """Step representing y = Pi_C(x) where C = nonneg orthant for all indices in ranges
        """
        super().__init__()
        self.x = x
        self.y = y

        if nonneg_ranges is list:
            self.nonneg_ranges = nonneg_ranges
        else:
            self.nonneg_ranges = [nonneg_ranges]

        # if real_ranges is list:
        #     self.real_ranges = real_ranges
        # else:
        #     self.real_ranges = [real_ranges]

        self._test_dims()
        self._compute_indices()

    def _test_dims(self):
        if self.x.dim != self.y.dim:
            raise AssertionError('iterate dimensions for partial nonneg proj do not match')

    def __str__(self):
        return f'{self.y.name} = PARTIAL_NONNEG_PROJ({self.x.name})\n\t nonneg_ranges={self.nonneg_ranges}'

    def _compute_indices(self):
        all_indices = set(range(self.x.dim))
        # real_indices = set()
        for range_bounds in self.nonneg_ranges:
            lo = range_bounds[0]
            hi = range_bounds[1]
            curr_range = set(range(lo, hi))
            all_indices = all_indices - curr_range
        real_indices = all_indices.copy()
        nonneg_indices = set(range(self.x.dim)) - real_indices
        self.nonneg_indices = nonneg_indices
        self.real_indices = real_indices

    # def process_ranges(self):
    #     n = self.x.dim
    #     all_indices = set(range(n))
    #     for range in all_indices:
    #         pass

    def get_output_var(self):
        return self.y

    def get_input_var(self):
        return self.x
