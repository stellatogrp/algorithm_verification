from algoverify.objectives.block_convergence_residual import BlockConvergenceResidual
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algoverify.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
from algoverify.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algoverify.high_level_alg_steps.linear_max_proj_step import LinearMaxProjStep
from algoverify.high_level_alg_steps.linear_step import LinearStep
from algoverify.init_set.box_set import BoxSet
from algoverify.init_set.const_shift_param_set import ConstShiftParamSet
from algoverify.init_set.l2_ball_set import L2BallSet
from algoverify.init_set.stack_set import StackSet
from algoverify.init_set.zero_set import ZeroSet
from algoverify.solvers.sdp_custom_solver.obj_canonicalizers.block_convergence_residual import \
    block_conv_resid_canon
from algoverify.solvers.sdp_custom_solver.obj_canonicalizers.convergence_residual import \
    conv_resid_canon
from algoverify.solvers.sdp_custom_solver.set_canonicalizers.box_set import (
    box_bound_canon, box_set_canon)
from algoverify.solvers.sdp_custom_solver.set_canonicalizers.const_shift_param_set import (
    const_shift_param_bound_canon, const_shift_param_set_canon)
from algoverify.solvers.sdp_custom_solver.set_canonicalizers.l2_ball_set import (
    l2_ball_bound_canon, l2_ball_set_canon)
from algoverify.solvers.sdp_custom_solver.set_canonicalizers.stack_set import (
    stack_set_canon, stack_bound_canon)
from algoverify.solvers.sdp_custom_solver.step_canonicalizers.linear_max_proj_step import (
    linear_max_proj_bound_canon, linear_max_proj_canon)
from algoverify.solvers.sdp_custom_solver.step_canonicalizers.linear_step import (
    linear_step_bound_canon, linear_step_canon)
from algoverify.solvers.sdp_custom_solver.step_canonicalizers.max_with_vec_step import (
    max_with_vec_bound_canon, max_with_vec_step_canon)
from algoverify.solvers.sdp_custom_solver.step_canonicalizers.min_with_vec_step import (
    min_with_vec_bound_canon, min_with_vec_step_canon)
from algoverify.solvers.sdp_custom_solver.step_canonicalizers.nonneg_orthant_proj_step import (
    nonneg_orthant_proj_bound_canon, nonneg_orthant_proj_canon)

HOLDER_SET_NORMS = {
    L2BallSet: 2,
}

OBJ_CANON_METHODS = {
    BlockConvergenceResidual: block_conv_resid_canon,
    ConvergenceResidual: conv_resid_canon,
}

SET_BOUND_CANON_METHODS = {
    BoxSet: box_bound_canon,
    ConstShiftParamSet: const_shift_param_bound_canon,
    L2BallSet: l2_ball_bound_canon,
    StackSet: stack_bound_canon,
    ZeroSet: l2_ball_bound_canon,
}

SET_CANON_METHODS = {
    BoxSet: box_set_canon,
    ConstShiftParamSet: const_shift_param_set_canon,
    L2BallSet: l2_ball_set_canon,
    StackSet: stack_set_canon,
    ZeroSet: l2_ball_set_canon,
}

STEP_BOUND_CANON_METHODS = {
    LinearMaxProjStep: linear_max_proj_bound_canon,
    LinearStep: linear_step_bound_canon,
    MaxWithVecStep: max_with_vec_bound_canon,
    MinWithVecStep: min_with_vec_bound_canon,
    NonNegProjStep: nonneg_orthant_proj_bound_canon,
}

STEP_CANON_METHODS = {
    LinearMaxProjStep: linear_max_proj_canon,
    LinearStep: linear_step_canon,
    MaxWithVecStep: max_with_vec_step_canon,
    MinWithVecStep: min_with_vec_step_canon,
    NonNegProjStep: nonneg_orthant_proj_canon,
}
