from algocert.solvers.global_solver.global_solver import GlobalSolver
from algocert.solvers.sdp_cgal_solver.sdp_cgal_solver import SDPCGALSolver
from algocert.solvers.sdp_custom_solver.sdp_custom_solver import SDPCustomSolver
from algocert.solvers.sdp_scgal_solver.sdp_scgal_solver import SDPSCGALSolver
from algocert.solvers.sdp_solver.sdp_solver import SDPSolver

SDP = "SDP"
GLOBAL = "GLOBAL"
SDP_CGAL = "SDP_CGAL"
SDP_CUSTOM = "SDP_CUSTOM"
SDP_SCGAL = "SDP_SCGAL"
DEFAULT = SDP

solver_mapping = {
    SDP: SDPSolver,
    SDP_CGAL: SDPCGALSolver,
    SDP_CUSTOM: SDPCustomSolver,
    SDP_SCGAL: SDPSCGALSolver,
    GLOBAL: GlobalSolver,
}
