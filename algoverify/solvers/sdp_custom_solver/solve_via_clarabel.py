import clarabel
import numpy as np
import scipy.sparse as spa


# The vec function as documented in api/cones
def vec(S):
    n = S.shape[0]
    S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n)] /= np.sqrt(2)
    return S[np.triu_indices(n)]


def spa_vec(S):
    return spa.csc_matrix(vec(S.todense()))


# The mat function as documented in api/cones
def mat(s):
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S


def spa_mat(s):
    return spa.csc_matrix(mat(s.todense()))


def solve_via_clarabel(C, A_vals, b_lvals, b_uvals, PSD_cones, problem_dim, handler):
    print('----solving via clarabel directly----')
    print('problem dim n:', problem_dim)
    c = vec(C.todense())
    x_dim = len(c)
    print('x_dim:', x_dim)

    Aeq = []
    Au = []
    Al = []

    beq = []
    bu = []
    bl = []

    Hp_mats = []
    bp_dims = []
    cone_dims = []


    for Ai, bli, bui in zip(A_vals, b_lvals, b_uvals):
        Ai_norm = spa.linalg.norm(Ai)
        # Ai_norm = 1
        if bli == bui:
            Aeq.append(spa_vec(Ai) / Ai_norm)
            beq.append(bli / Ai_norm)
        else:
            if bui < np.inf:
                Au.append(spa_vec(Ai) / Ai_norm)
                bu.append(bui / Ai_norm)
            if bli > -np.inf:
                Al.append(-spa_vec(Ai) / Ai_norm)
                bl.append(-bli / Ai_norm)

    for cone in PSD_cones:
        # H = cone.get_Hsymm_mat(problem_dim)
        H = cone.get_sparse_Hsymm_mat(problem_dim)
        cone_dim = int((np.sqrt(8 * H.shape[0] + 1) - 1) / 2)

        Hp_mats.append(-H)
        bp_dims.append(H.shape[0])
        cone_dims.append(cone_dim)
    zero_cone_dim = len(Aeq)
    nonneg_cone_dim = len(Au) + len(Al)
    # A = spa.vstack(Aeq + Au + Al, format='csc')
    A = spa.vstack(Aeq + Au + Al + Hp_mats, format='csc')
    b = np.array(beq + bu + bl)
    b = np.hstack([b, np.zeros(np.sum(bp_dims))])

    cones = [
        clarabel.ZeroConeT(zero_cone_dim),
        clarabel.NonnegativeConeT(nonneg_cone_dim),
    ]
    for cone_dim in cone_dims:
        cones.append(clarabel.PSDTriangleConeT(cone_dim))

    print(cones)
    P = spa.csc_matrix((x_dim, x_dim))
    settings = clarabel.DefaultSettings()

    solver = clarabel.DefaultSolver(P, c, A, b, cones, settings)
    solver.solve()

    exit(0)
