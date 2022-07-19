import numpy as np
import scipy.sparse as spa

from gurobipy import GRB


def conv_resid_canon(iterate, model, iterate_to_gp_var_map):
    x = iterate.get_iterate()
    n = x.get_dim()

    x_var = iterate_to_gp_var_map[x]
    xN = x_var[-1]
    xNminus1 = x_var[-2]

    In = spa.eye(n)
    twoIn = 2 * In

    obj = xN @ In @ xN - xN @ twoIn @ xNminus1 + xNminus1 @ In @ xNminus1
    model.setObjective(obj, GRB.MAXIMIZE)
