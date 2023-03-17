from algocert.solvers.global_solver.global_solver import GlobalSolver
from algocert.solvers.sdp_cgal_solver.sdp_cgal_solver import SDPCGALSolver
from algocert.solvers.sdp_solver.sdp_solver import SDPSolver

SDP = "SDP"
GLOBAL = "GLOBAL"
SDP_CGAL = "SDP_CGAL"
DEFAULT = SDP

solver_mapping = {
    SDP: SDPSolver,
    SDP_CGAL: SDPCGALSolver,
    GLOBAL: GlobalSolver,
}
