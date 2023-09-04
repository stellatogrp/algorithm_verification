import numpy as np

from algocert.basic_algorithm_steps.step import Step
from algocert.variables.iterate import Iterate


class NonNegProjStep(Step):

    """Docstring for NonNegStep. """

    def __init__(self, y: Iterate, x: Iterate):
        """Step representing y = (x)_+
        """
        super().__init__()
        self.x = x
        self.y = y
        self._test_dims()

    def _test_dims(self):
        if self.x.dim != self.y.dim:
            raise AssertionError('iterate dimensions for nonneg proj do not match')

    def __str__(self):
        return f'{self.y.name} = NONNEG_PROJ({self.x.name})'

    def get_output_var(self):
        return self.y

    def get_input_var(self):
        return self.x

    #  def apply(self, x):
    #      return intermediate

    def apply(self, k, iter_to_id_map, ranges, out):
        y = self.y
        x = self.x
        # u_vec = []
        # for x in self.u:
        #     if not x.is_param:
        #         if iter_to_id_map[y] <= iter_to_id_map[x]:
        #             x_range = ranges[k-1][x]
        #         else:
        #             x_range = ranges[k][x]
        #     else:
        #         x_range = ranges[x]
        #     u_vec.append(out[x_range[0]: x_range[1]])
        # # print(np.vstack(u_vec))
        # return self.A @ np.vstack(u_vec)
        if iter_to_id_map[y] <= iter_to_id_map[x]:
            x_range = ranges[k-1][x]
        else:
            x_range = ranges[k][x]
        return np.maximum(out[x_range[0]: x_range[1]], 0)
