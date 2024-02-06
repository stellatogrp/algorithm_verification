from algoverify.high_level_alg_steps.nonneg_lin_step import NonNegLinStep
from algoverify.init_set.box_set import BoxSet
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.solvers.sdp_scgal_solver.obj_canonicalizers.convergence_residual import \
    conv_resid_canon
from algoverify.solvers.sdp_scgal_solver.set_canonicalizers.box_set import (
    box_set_preprocess, box_set_primitive_2)
from algoverify.solvers.sdp_scgal_solver.step_canonicalizers.nonneg_lin_step import (
    nonneg_lin_preprocess, nonneg_lin_primitive_2)

SET_PREPROCESS_METHODS = {
    BoxSet: box_set_preprocess,
}

STEP_PREPROCESS_METHODS = {
    NonNegLinStep: nonneg_lin_preprocess,
}

OBJ_CANON_METHODS = {
    ConvergenceResidual: conv_resid_canon,
}

SET_PRIMITIVE_2_METHODS = {
    BoxSet: box_set_primitive_2,
}

STEP_PRIMITIVE_2_METHODS = {
    NonNegLinStep: nonneg_lin_primitive_2,
}
