from algoverify.basic_algorithm_steps.basic_linear_step import BasicLinearStep
from algoverify.basic_algorithm_steps.block_step import BlockStep
from algoverify.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algoverify.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
from algoverify.basic_algorithm_steps.nonneg_orthant_proj_step import \
    NonNegProjStep
from algoverify.high_level_alg_steps.box_proj_step import BoxProjStep
from algoverify.high_level_alg_steps.linear_step import LinearStep
from algoverify.high_level_alg_steps.nonneg_lin_step import NonNegLinStep
from algoverify.init_set.box_set import BoxSet
from algoverify.init_set.centered_l2_ball_set import CenteredL2BallSet
from algoverify.init_set.const_set import ConstSet
from algoverify.init_set.ellipsoidal_set import EllipsoidalSet
from algoverify.init_set.linf_ball_set import LInfBallSet
from algoverify.init_set.vec_span_set import VecSpanSet
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.objectives.l1_conv_resid import L1ConvResid
from algoverify.objectives.outer_prod_trace import OuterProdTrace
from algoverify.solvers.sdp_solver.obj_canonicalizer.convergence_residual import \
    conv_resid_canon
from algoverify.solvers.sdp_solver.obj_canonicalizer.l1_conv_resid import \
    l1_conv_resid_canon
from algoverify.solvers.sdp_solver.obj_canonicalizer.outer_prod_trace import \
    outer_prod_trace_canon
from algoverify.solvers.sdp_solver.set_canonicalizers.box_set import (
    box_bound_canon, box_canon)
from algoverify.solvers.sdp_solver.set_canonicalizers.centered_l2_ball_set import \
    centered_l2_ball_canon
from algoverify.solvers.sdp_solver.set_canonicalizers.const_set import (
    const_bound_canon, const_canon)
from algoverify.solvers.sdp_solver.set_canonicalizers.ellipsoidal_set import \
    ellipsoidal_canon
from algoverify.solvers.sdp_solver.set_canonicalizers.linf_ball_set import \
    linf_ball_canon
from algoverify.solvers.sdp_solver.set_canonicalizers.vec_span_set import \
    vec_span_canon
from algoverify.solvers.sdp_solver.step_canonicalizers.basic_linear_step import (
    basic_linear_step_bound_canon, basic_linear_step_canon)
from algoverify.solvers.sdp_solver.step_canonicalizers.block_step import (
    block_step_bound_canon, block_step_canon)
from algoverify.solvers.sdp_solver.step_canonicalizers.box_proj_step import \
    box_proj_step_canon
from algoverify.solvers.sdp_solver.step_canonicalizers.linear_step import (
    linear_step_bound_canon, linear_step_canon)
from algoverify.solvers.sdp_solver.step_canonicalizers.max_with_vec_step import (
    max_vec_bound_canon, max_vec_canon)
from algoverify.solvers.sdp_solver.step_canonicalizers.min_with_vec_step import (
    min_vec_bound_canon, min_vec_canon)
from algoverify.solvers.sdp_solver.step_canonicalizers.nonneg_lin_step import \
    nonneg_lin_canon
from algoverify.solvers.sdp_solver.step_canonicalizers.nonneg_orthant_proj_step import (
    nonneg_orthant_proj_bound_canon, nonneg_orthant_proj_canon)

SET_CANON_METHODS = {
    CenteredL2BallSet: centered_l2_ball_canon,
    EllipsoidalSet: ellipsoidal_canon,
    BoxSet: box_canon,
    ConstSet: const_canon,
    LInfBallSet: linf_ball_canon,
    VecSpanSet: vec_span_canon,
}

STEP_CANON_METHODS = {
    BlockStep: block_step_canon,
    BasicLinearStep: basic_linear_step_canon,
    LinearStep: linear_step_canon,
    NonNegLinStep: nonneg_lin_canon,
    NonNegProjStep: nonneg_orthant_proj_canon,
    MaxWithVecStep: max_vec_canon,
    MinWithVecStep: min_vec_canon,
}

RLT_CANON_SET_METHODS = {
    BoxSet: box_bound_canon,
    ConstSet: const_bound_canon,
}

RLT_CANON_STEP_METHODS = {
    BlockStep: block_step_bound_canon,
    BasicLinearStep: basic_linear_step_bound_canon,
    LinearStep: linear_step_bound_canon,
    NonNegProjStep: nonneg_orthant_proj_bound_canon,
    MaxWithVecStep: max_vec_bound_canon,
    MinWithVecStep: min_vec_bound_canon,
}

OBJ_CANON_METHODS = {
    ConvergenceResidual: conv_resid_canon,
    L1ConvResid: l1_conv_resid_canon,
    OuterProdTrace: outer_prod_trace_canon,
}

HL_TO_BASIC_STEP_METHODS = {
    # LinearStep: linear_step_canon,
    BoxProjStep: box_proj_step_canon,
}
