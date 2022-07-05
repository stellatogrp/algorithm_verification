import cvxpy as cp
import numpy as np

from certification_problem.solvers.sdp_solver.set_canonicalizers.box_set import box_canon


def linf_ball_canon(init_set, handler):
    return box_canon(init_set, handler)
