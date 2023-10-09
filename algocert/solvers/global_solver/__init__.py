from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import \
    NonNegProjStep
from algocert.basic_algorithm_steps.partial_nonneg_orthant_proj_step import \
    PartialNonNegProjStep
from algocert.high_level_alg_steps.linear_step import LinearStep
from algocert.high_level_alg_steps.nonneg_lin_step import NonNegLinStep
from algocert.init_set.affine_vec_set import AffineVecSet
from algocert.init_set.box_set import BoxSet
from algocert.init_set.box_stack_set import BoxStackSet
from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
from algocert.init_set.const_set import ConstSet
from algocert.init_set.control_example_set import ControlExampleSet
from algocert.init_set.ellipsoidal_set import EllipsoidalSet
from algocert.init_set.linf_ball_set import LInfBallSet
from algocert.init_set.l2_ball_set import L2BallSet
from algocert.init_set.stack_set import StackSet
from algocert.init_set.vec_span_set import VecSpanSet
from algocert.init_set.zero_set import ZeroSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.objectives.l1_conv_resid import L1ConvResid
from algocert.objectives.lin_comb_squared_norm import LinCombSquaredNorm
from algocert.objectives.linf_conv_resid import LInfConvResid
from algocert.objectives.outer_prod_trace import OuterProdTrace
from algocert.solvers.global_solver.obj_canonicalizers.convergence_residual import \
    conv_resid_canon
from algocert.solvers.global_solver.obj_canonicalizers.l1_conv_resid import \
    l1_conv_resid_canon
from algocert.solvers.global_solver.obj_canonicalizers.lin_comb_squared_norm import \
    lin_comb_squared_norm_canon
from algocert.solvers.global_solver.obj_canonicalizers.linf_conv_resid import \
    linf_conv_resid_canon
from algocert.solvers.global_solver.obj_canonicalizers.outer_prod_trace import \
    outer_prod_trace_canon
from algocert.solvers.global_solver.set_canonicalizers.affine_vec_set import \
    affine_vec_set_canon
from algocert.solvers.global_solver.set_canonicalizers.box_set import (
    box_set_bound_canon, box_set_canon)
from algocert.solvers.global_solver.set_canonicalizers.box_stack_set import (
    box_stack_set_bound_canon, box_stack_set_canon)
from algocert.solvers.global_solver.set_canonicalizers.centered_l2_ball_set import \
    centered_l2_ball_canon
from algocert.solvers.global_solver.set_canonicalizers.const_set import (
    const_set_bound_canon, const_set_canon)
from algocert.solvers.global_solver.set_canonicalizers.control_example_set import (
    control_example_set_bound_canon, control_example_set_canon)
from algocert.solvers.global_solver.set_canonicalizers.ellipsoidal_set import \
    ellipsoidal_set_canon
from algocert.solvers.global_solver.set_canonicalizers.linf_ball_set import \
    linf_ball_set_canon
from algocert.solvers.global_solver.set_canonicalizers.l2_ball_set import (
    l2_ball_set_bound_canon, l2_ball_set_canon)
from algocert.solvers.global_solver.set_canonicalizers.stack_set import (
    stack_set_bound_canon, stack_set_canon)
from algocert.solvers.global_solver.set_canonicalizers.vec_span_set import \
    vec_span_set_canon
from algocert.solvers.global_solver.step_canonicalizers.linear_step import (
    linear_step_bound_canon, linear_step_canon)
from algocert.solvers.global_solver.step_canonicalizers.max_with_vec_step import (
    max_vec_bound_canon, max_vec_canon)
from algocert.solvers.global_solver.step_canonicalizers.min_with_vec_step import (
    min_vec_bound_canon, min_vec_canon)
from algocert.solvers.global_solver.step_canonicalizers.nonneg_lin_step import (
    nonneg_lin_bound_canon, nonneg_lin_canon)
from algocert.solvers.global_solver.step_canonicalizers.nonneg_orthant_proj_step import (
    nonneg_orthant_proj_bound_canon, nonneg_orthant_proj_canon)
from algocert.solvers.global_solver.step_canonicalizers.partial_nonneg_orthant_proj_step import (
    partial_nonneg_orthant_proj_bound_canon, partial_nonneg_orthant_proj_canon)

SET_CANON_METHODS = {
    CenteredL2BallSet: centered_l2_ball_canon,
    EllipsoidalSet: ellipsoidal_set_canon,
    BoxSet: box_set_canon,
    BoxStackSet: box_stack_set_canon,
    ConstSet: const_set_canon,
    ControlExampleSet: control_example_set_canon,
    LInfBallSet: linf_ball_set_canon,
    L2BallSet: l2_ball_set_canon,
    StackSet: stack_set_canon,
    VecSpanSet: vec_span_set_canon,
    ZeroSet: l2_ball_set_canon,
}

BOUND_SET_CANON_METHODS = {
    BoxSet: box_set_bound_canon,
    BoxStackSet: box_stack_set_bound_canon,
    ConstSet: const_set_bound_canon,
    ControlExampleSet: control_example_set_bound_canon,
    L2BallSet: l2_ball_set_bound_canon,
    StackSet: stack_set_bound_canon,
    ZeroSet: l2_ball_set_bound_canon,
}

STEP_CANON_METHODS = {
    LinearStep: linear_step_canon,
    MaxWithVecStep: max_vec_canon,
    MinWithVecStep: min_vec_canon,
    NonNegLinStep: nonneg_lin_canon,
    NonNegProjStep: nonneg_orthant_proj_canon,
    PartialNonNegProjStep: partial_nonneg_orthant_proj_canon,
}

BOUND_STEP_CANON_METHODS = {
    LinearStep: linear_step_bound_canon,
    MaxWithVecStep: max_vec_bound_canon,
    MinWithVecStep: min_vec_bound_canon,
    NonNegLinStep: nonneg_lin_bound_canon,
    NonNegProjStep: nonneg_orthant_proj_bound_canon,
    PartialNonNegProjStep: partial_nonneg_orthant_proj_bound_canon,
}

OBJ_CANON_METHODS = {
    ConvergenceResidual: conv_resid_canon,
    OuterProdTrace: outer_prod_trace_canon,
    L1ConvResid: l1_conv_resid_canon,
    LInfConvResid: linf_conv_resid_canon,
    LinCombSquaredNorm: lin_comb_squared_norm_canon,
}
