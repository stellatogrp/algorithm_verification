from certification_problem.init_set.centered_l2_ball_set import CenteredL2BallSet
from certification_problem.init_set.ellipsoidal_set import EllipsoidalSet
from certification_problem.init_set.box_set import BoxSet
from certification_problem.init_set.const_set import ConstSet
from certification_problem.init_set.linf_ball_set import LInfBallSet
from certification_problem.init_set.vec_span_set import VecSpanSet

from certification_problem.solvers.global_solver.set_canonicalizers.centered_l2_ball_set import centered_l2_ball_canon
from certification_problem.solvers.global_solver.set_canonicalizers.ellipsoidal_set import ellipsoidal_set_canon
from certification_problem.solvers.global_solver.set_canonicalizers.box_set import (
    box_set_canon, box_set_bound_canon, )
from certification_problem.solvers.global_solver.set_canonicalizers.const_set import (
    const_set_canon, const_set_bound_canon, )
from certification_problem.solvers.global_solver.set_canonicalizers.linf_ball_set import linf_ball_set_canon
from certification_problem.solvers.global_solver.set_canonicalizers.vec_span_set import vec_span_set_canon

from certification_problem.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from certification_problem.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from certification_problem.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from certification_problem.basic_algorithm_steps.min_with_vec_step import MinWithVecStep

from certification_problem.solvers.global_solver.step_canonicalizers.hl_linear_step import hl_linear_step_canon
from certification_problem.solvers.global_solver.step_canonicalizers.nonneg_orthant_proj_step import (
    nonneg_orthant_proj_canon, )
from certification_problem.solvers.global_solver.step_canonicalizers.max_with_vec_step import (
    max_vec_canon, )
from certification_problem.solvers.global_solver.step_canonicalizers.min_with_vec_step import (
    min_vec_canon, )

from certification_problem.objectives.convergence_residual import ConvergenceResidual
from certification_problem.objectives.outer_prod_trace import OuterProdTrace
from certification_problem.objectives.linf_conv_resid import LInfConvResid

from certification_problem.solvers.global_solver.obj_canonicalizers.convergence_residual import conv_resid_canon
from certification_problem.solvers.global_solver.obj_canonicalizers.outer_prod_trace import outer_prod_trace_canon
from certification_problem.solvers.global_solver.obj_canonicalizers.linf_conv_resid import linf_conv_resid_canon

SET_CANON_METHODS = {
    CenteredL2BallSet: centered_l2_ball_canon,
    EllipsoidalSet: ellipsoidal_set_canon,
    BoxSet: box_set_canon,
    ConstSet: const_set_canon,
    LInfBallSet: linf_ball_set_canon,
    VecSpanSet: vec_span_set_canon,
}

BOUND_SET_CANON_METHODS = {
    BoxSet: box_set_bound_canon,
    ConstSet: const_set_bound_canon,
}

STEP_CANON_METHODS = {
    HighLevelLinearStep: hl_linear_step_canon,
    NonNegProjStep: nonneg_orthant_proj_canon,
    MaxWithVecStep: max_vec_canon,
    MinWithVecStep: min_vec_canon,
}

OBJ_CANON_METHODS = {
    ConvergenceResidual: conv_resid_canon,
    OuterProdTrace: outer_prod_trace_canon,
    LInfConvResid: linf_conv_resid_canon,
}
