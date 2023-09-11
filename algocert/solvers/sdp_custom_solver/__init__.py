from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.solvers.sdp_custom_solver.obj_canonicalizers.convergence_residual import \
    conv_resid_canon

OBJ_CANON_METHODS = {
    ConvergenceResidual: conv_resid_canon,
}
