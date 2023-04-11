from algocert.basic_algorithm_steps.block_step import BlockStep
from algocert.basic_algorithm_steps.linear_step import LinearStep
from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import \
    NonNegProjStep
from algocert.high_level_alg_steps.box_proj_step import BoxProjStep
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.high_level_alg_steps.nonneg_lin_step import NonNegLinStep
from algocert.init_set.box_set import BoxSet
from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.solvers.sdp_cgal_solver.obj_canonicalizers.convergence_residual import \
    conv_resid_canon
from algocert.solvers.sdp_cgal_solver.set_canonicalizers.box_set import \
    box_set_canon
from algocert.solvers.sdp_cgal_solver.set_canonicalizers.centered_l2_ball_set import \
    centered_l2_ball_canon
from algocert.solvers.sdp_cgal_solver.step_canonicalizers.hl_linear_step import \
    hl_linear_step_canon
from algocert.solvers.sdp_cgal_solver.step_canonicalizers.nonneg_lin_step import \
    nonneg_lin_canon
from algocert.solvers.sdp_cgal_solver.step_canonicalizers.nonneg_orthant_proj_step import \
    nonneg_orthant_proj_canon

HL_TO_BASIC_STEP_METHODS = {
    HighLevelLinearStep: hl_linear_step_canon,
}

OBJ_CANON_METHODS = {
    ConvergenceResidual: conv_resid_canon,
}

SET_CANON_METHODS = {
    BoxSet: box_set_canon,
    CenteredL2BallSet: centered_l2_ball_canon,
}

STEP_CANON_METHODS = {
    HighLevelLinearStep: hl_linear_step_canon,
    NonNegLinStep: nonneg_lin_canon,
    NonNegProjStep: nonneg_orthant_proj_canon,
}
