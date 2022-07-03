from certification_problem.init_set.l2_ball import L2BallSet

from certification_problem.solvers.sdp_solver.set_canonicalizers.l2_ball import l2_ball_canon

from certification_problem.algorithm_steps.block_step import BlockStep
from certification_problem.algorithm_steps.linear_step import LinearStep
from certification_problem.algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep

from certification_problem.solvers.sdp_solver.step_canonicalizers.block_step import block_step_canon
from certification_problem.solvers.sdp_solver.step_canonicalizers.linear_step import linear_step_canon
from certification_problem.solvers.sdp_solver.step_canonicalizers.nonneg_orthant_proj_step import (
    nonneg_orthant_proj_canon, )

SET_CANON_METHODS = {
    L2BallSet: l2_ball_canon,
}

STEP_CANON_METHODS = {
    BlockStep: block_step_canon,
    LinearStep: linear_step_canon,
    NonNegProjStep: nonneg_orthant_proj_canon,
}
