import cvxpy as cp
import numpy as np

from certification_problem.solvers.global_solver.set_canonicalizers.box_set import box_set_canon


def linf_ball_canon(init_set, model, var_to_gp_var_map):
    return box_set_canon(init_set, model, var_to_gp_var_map)
