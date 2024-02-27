import cvxpy as cp
import numpy as np


def vec_span_canon(init_set, handler):
    x = init_set.get_iterate()
    v = init_set.v
    a = init_set.a
    b = init_set.b
    x_var = handler.iterate_vars[x].get_cp_var()
    xxT_var = handler.iterate_outerproduct_vars[x]

    c = cp.Variable()  # TODO: figure out, do we need to include these extra vars in the return?
    c_squared = cp.Variable()
    init_set.set_c_vars(c, c_squared)

    return [
        a <= c <= b,
        a ** 2 <= c_squared <= b ** 2,
        x_var == c * v,
        xxT_var == c_squared * v @ v.T,
        cp.bmat([
            [xxT_var, x_var],
            [x_var.T, np.array([[1]])]
        ]) >> 0,
        cp.bmat([
            [c_squared, c],
            [c, 1]
        ]) >> 0,
    ]
