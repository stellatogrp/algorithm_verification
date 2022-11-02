from algocert.basic_algorithm_steps.block_step import BlockStep
from algocert.basic_algorithm_steps.linear_step import LinearStep
from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
from algocert.high_level_alg_steps.box_proj_step import BoxProjStep
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.init_set.box_set import BoxSet
from algocert.solvers.sdp_admm_solver.step_canonicalizers.hl_linear_step import \
    hl_linear_step_canon

HL_TO_BASIC_STEP_METHODS = {
    HighLevelLinearStep: hl_linear_step_canon,
}
