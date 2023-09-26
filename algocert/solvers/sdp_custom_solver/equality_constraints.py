import numpy as np
import scipy.sparse as spa

from algocert.solvers.sdp_custom_solver.range_handler import RangeHandler1D, RangeHandler2D
from algocert.solvers.sdp_custom_solver.utils import map_linstep_to_ranges


def equality1D_constraints(D, y, A, u, b, k, handler):
    '''
        Constraints for D^k = Au + b where u = [u1, u2, ..., ul]
    '''
    problem_dim = handler.problem_dim
    iter_bound_map = handler.iter_bound_map

    n = y.get_dim()
    yrange = iter_bound_map[y][k]
    uranges = map_linstep_to_ranges(y, u, k, handler)

    yrange_handler = RangeHandler1D(yrange)
    urange_handler = RangeHandler1D(uranges)
    # print(yrange_handler.index_matrix(), urange_handler.index_matrix())

    A_matrices = []
    b_lvals = []
    b_uvals = []

    for i in range(n):
        outmat = np.zeros((problem_dim, problem_dim))
        Di = D.todense()[i]
        Ai = A.todense()[i]
        # print(Di, Ai)
        outmat[yrange_handler.index_matrix()] = Di.T
        outmat[urange_handler.index_matrix()] = -Ai.T
        # print(outmat)
        outmat = (outmat + outmat.T) / 2
        # print(outmat)
        A_matrices.append(spa.csc_matrix(outmat))
        b_lvals.append(b[i, 0])
        b_uvals.append(b[i, 0])

    return A_matrices, b_lvals, b_uvals


def equality2D_constraints(D, y, A, u, b, k, handler):

    problem_dim = handler.problem_dim
    iter_bound_map = handler.iter_bound_map

    n = y.get_dim()
    yrange = iter_bound_map[y][k]
    uranges = map_linstep_to_ranges(y, u, k, handler)

    RangeHandler1D(yrange)
    yrange2D_handler = RangeHandler2D(yrange, yrange)
    urange1D_handler = RangeHandler1D(uranges)
    urange2D_handler = RangeHandler2D(uranges, uranges)

    # print(yrange2D_handler.index_matrix(), urange2D_handler.index_matrix())

    A_matrices = []
    b_lvals = []
    b_uvals = []
    psd_cone_handlers = []

    # psd_cone_handlers += [PSDConeHandler([yrange] + uranges)]

    n = y.get_dim()

    bbT = np.outer(b, b)

    D = D.todense()
    A = A.todense()
    for i in range(n):
        for j in range(i, n):
            outmat = np.zeros((problem_dim, problem_dim))
            Di = D[i].T.reshape((-1, 1))
            DTj = D.T[:, j].T.reshape((1, -1))
            # print(Di.shape, DTj.shape)
            Ai = A[i].T.reshape((-1, 1))
            ATj = A.T[:, j].T.reshape((1, -1))
            # print(Ai.shape, ATj.shape)

            DiDTj = Di @ DTj
            AiATj = Ai @ ATj

            outmat[yrange2D_handler.index_matrix()] = DiDTj
            outmat[urange2D_handler.index_matrix()] = -AiATj
            outmat[urange1D_handler.index_matrix()] = -Ai * b[j, 0]
            # print(urange1D_handler.index_matrix_horiz())
            outmat[urange1D_handler.index_matrix_horiz()] = -b[i, 0] * ATj

            outmat = (outmat + outmat.T) / 2
            outmat = spa.csc_matrix(outmat)
            A_matrices.append(outmat)
            b_lvals.append(bbT[i, j])
            b_uvals.append(bbT[i, j])


    return A_matrices, b_lvals, b_uvals, psd_cone_handlers


def curr_or_prev(var1, var2, k, iter_id_map):
    """
    Returning which step of var2 to use
    I.e. if y = LinStep(x), need to know if y^{k} depends on x^k or x^{k-1}
    """
    i1 = iter_id_map[var1]
    i2 = iter_id_map[var2]
    if i1 <= i2:
        return k-1
    return k
