from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algocert.high_level_alg_steps.linear_step import LinearStep
from algocert.init_set.box_set import BoxSet
from algocert.solvers.sdp_custom_solver.obj_canonicalizers.convergence_residual import \
    conv_resid_canon
from algocert.solvers.sdp_custom_solver.set_canonicalizers.box_set import (
    box_bound_canon, box_set_canon)
from algocert.solvers.sdp_custom_solver.step_canonicalizers.linear_step import (
    linear_step_bound_canon, linear_step_canon)
from algocert.solvers.sdp_custom_solver.step_canonicalizers.nonneg_orthant_proj_step import (
    nonneg_orthant_proj_bound_canon, nonneg_orthant_proj_canon)

OBJ_CANON_METHODS = {
    ConvergenceResidual: conv_resid_canon,
}

SET_BOUND_CANON_METHODS = {
    BoxSet: box_bound_canon,
}

SET_CANON_METHODS = {
    BoxSet: box_set_canon,
}

STEP_BOUND_CANON_METHODS = {
    LinearStep: linear_step_bound_canon,
    NonNegProjStep: nonneg_orthant_proj_bound_canon,
}

STEP_CANON_METHODS = {
    LinearStep: linear_step_canon,
    NonNegProjStep: nonneg_orthant_proj_canon,
}
