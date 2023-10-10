import cvxpy as cp
import numpy as np
import scipy.sparse as spa
import scs


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


def solve_via_scs_single(C, A_vals, b_lvals, b_uvals, PSD_cones, problem_dim):
    print('solving lp via scs directly')
    c = vec(C.todense())
    x_dim = len(c)
    print('problem_dim', problem_dim, 'x_dim:', x_dim)
    A = [spa.csc_matrix((1, x_dim))]
    print(len(A_vals), len(b_lvals), len(b_uvals))
    for A_curr in A_vals:
        # print(spa_vec(A_curr))
        A.append(spa_vec(A_curr))
    print('num boxes:', len(A))
    print('num psd cones:', len(PSD_cones))
    # for cone in PSD_cones:
    #     # H = cone.get_Hsymm_mat(problem_dim)
    #     H = cone.get_sparse_Hsymm_mat(problem_dim)
    #     print(H.shape)
    #     # print(cone.row_indices)
    #     A.append(H)
    #     cone_dims.append(H.shape[0])
        # print(H)
    # box_dim = len(A_vals)
    # print(box_dim)

    # remember to negate A for the cones
    A = -spa.vstack(A, format='csc')
    print(A.shape, len(b_lvals), len(b_uvals))
    b = np.zeros(A.shape[0])
    b[0] = 1
    print(b.shape)

    A = spa.vstack([A, -spa.eye(x_dim)], format='csc')
    b = np.hstack([b, np.zeros(x_dim)])

    data = dict(A=A, b=b, c=c)
    cone = dict(bu=b_uvals, bl=b_lvals, s=problem_dim)
    solver = scs.SCS(data, cone, eps_abs=1e-5, eps_rel=1e-5, max_iters=int(1e4))
    solver.solve()
    # print(cone_dims)

    exit(0)


def solve_via_scs_split(C, A_vals, b_lvals, b_uvals, PSD_cones, problem_dim):
    print('----solving via scs directly----')
    print('problem dim n:', problem_dim)
    c = vec(C.todense())
    x_dim = len(c)
    print('x_dim:', x_dim)
    A = [spa.csc_matrix((1, x_dim))]
    print(len(A_vals), len(b_lvals), len(b_uvals))
    for A_curr in A_vals:
        # print(spa_vec(A_curr))
        A.append(spa_vec(A_curr))
    print('num boxes:', len(A))
    print('num psd cones:', len(PSD_cones))
    cone_dims = []
    for cone in PSD_cones:
        # H = cone.get_Hsymm_mat(problem_dim)
        H = cone.get_sparse_Hsymm_mat(problem_dim)
        print(H.shape)
        # print(cone.row_indices)
        A.append(H)
        cone_dim = int((np.sqrt(8 * H.shape[0] + 1) - 1) / 2)
        cone_dims.append(cone_dim)
        # print(H)
    # box_dim = len(A_vals)
    # print(box_dim)

    # remember to negate A for the cones
    A = -spa.vstack(A, format='csc')
    print(A.shape)
    print(cone_dims)

    b_dim = A.shape[0]
    b = np.zeros(b_dim)
    b[0] = 1
    # b_lower = np.hstack([1, b_lvals])
    # b_upper = np.hstack([])
    # print(len(b))
    data = dict(A=A, b=b, c=c)
    # cone = dict(bu=b_uvals, bl=b_lvals, s=np.array(cone_dims))
    cone = dict(bu=b_uvals, bl=b_lvals, s=cone_dims)
    print(len(b_uvals) + np.sum(cone_dims), len(b_lvals) + np.sum(cone_dims), len(b))

    print(A.shape, b.shape, c.shape)
    solver = scs.SCS(data, cone, eps_abs=1e-5, eps_rel=1e-5)
    solver.solve()

    exit(0)
    return 0, 0


def solve_via_cvxpy_lp(C, A_vals, b_lvals, b_uvals, PSD_cones, problem_dim):
    print('checking the LP')
    n = C.shape[0]
    X = cp.Variable((n, n), symmetric=True)
    obj = cp.trace(C @ X)

    constraints = []
    for Ai, bli, bui in zip(A_vals, b_lvals, b_uvals):
        if bli > -np.inf:
            constraints += [cp.trace(Ai @ X) >= bli]
        if bui < np.inf:
            constraints += [cp.trace(Ai @ X) <= bui]
    # constraints += [X >> 0]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    res = prob.solve()
    print(res)

    print('checking the vectorized LP')
    constraints = []
    n = int((C.shape[0] * (C.shape[0] + 1)) / 2)
    print(n)
    x = cp.Variable(n)
    obj = spa_vec(C) @ x

    for Ai, bli, bui in zip(A_vals, b_lvals, b_uvals):
        if bli > -np.inf:
            constraints += [spa_vec(Ai) @ x >= bli]
        if bui < np.inf:
            constraints += [spa_vec(Ai) @ x <= bui]

    prob = cp.Problem(cp.Minimize(obj), constraints)
    res = prob.solve()
    print(res)

    exit(0)
    return 0, 0

def solve_via_scs_nosdp(C, A_vals, b_lvals, b_uvals, PSD_cones, problem_dim):
    print('----solving via scs directly----')
    print('problem dim n:', problem_dim)
    c = vec(C.todense())
    x_dim = len(c)
    print('x_dim:', x_dim)

    A = []

    print(len(A_vals), len(b_lvals), len(b_uvals))
    for A_curr in A_vals:
        A.append(spa_vec(A_curr))

    A = spa.vstack(A, format='csc')
    b_lower = np.array(b_lvals)
    b_upper = np.array(b_uvals)
    print(A.shape)
    print(type(b_lower), b_lower.shape)

    A = spa.vstack([
        spa.csc_matrix((1, x_dim)),
        -A,
    ], format='csc')

    b = np.zeros(A.shape[0])
    b[0] = 1
    data = dict(A=A, b=b, c=c)
    cones = dict(bu=b_upper, bl=b_lower)
    solver = scs.SCS(data, cones, eps_abs=1e-5, eps_rel=1e-5)

    solver.solve()

    exit(0)
    return 0, 0


def sample_x(handler):
    # print(handler.var_lowerbounds)
    # exit(0)
    l = handler.var_lowerbounds
    return vec(l @ l.T)


def solve_via_scs(C, A_vals, b_lvals, b_uvals, PSD_cones, problem_dim, handler):
    print('----solving via scs directly----')
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
        # print(Ai_norm)

    for cone in PSD_cones:
        # H = cone.get_Hsymm_mat(problem_dim)
        H = cone.get_sparse_Hsymm_mat(problem_dim)
        # print(H.shape)
        # print(cone.row_indices)
        # A.append(H)
        cone_dim = int((np.sqrt(8 * H.shape[0] + 1) - 1) / 2)
        # cone_dims.append(cone_dim)
        # print(H.shape[0], cone_dim)

        Hp_mats.append(-H)
        bp_dims.append(H.shape[0])
        cone_dims.append(cone_dim)
    zero_cone_dim = len(Aeq)
    nonneg_cone_dim = len(Au) + len(Al)

    # A = spa.vstack(Aeq + Au + Al, format='csc')
    A = spa.vstack(Aeq + Au + Al + Hp_mats, format='csc')
    b = np.array(beq + bu + bl)
    b = np.hstack([b, np.zeros(np.sum(bp_dims))])

    data = dict(A=A, b=b, c=c)
    # cones = dict(z=zero_cone_dim, l=nonneg_cone_dim)
    cones = dict(z=zero_cone_dim, l=nonneg_cone_dim, s=cone_dims)
    solver = scs.SCS(data,
                    cones,
                    eps_abs=1e-4,
                    eps_rel=1e-4,
                    max_iters=int(1e6),
                    use_indirect=True,
                    acceleration_lookback=0,
                    )
    # xfull_ws = np.zeros(x_dim)
    # xpart_ws = sample_x(handler)
    # # print(A.shape, x_dim, xpart_ws.shape)
    # xfull_ws[:xpart_ws.shape[0]] = xpart_ws
    # # print(xfull_ws)
    # # exit(0)
    # # sol = solver.solve(warm_start=True, x=xfull_ws)
    # print(handler.var_warmstart)
    x_ws = handler.var_warmstart
    mat_X_ws = np.bmat([
        [x_ws @ x_ws.T, x_ws],
        [x_ws.T, np.array([[1]])]
    ])
    mat_X_ws = vec(mat_X_ws)
    # print(mat_X_ws.shape)
    s_ws = b - A @ mat_X_ws
    # print(s_ws.shape)
    # exit(0)
    sol = solver.solve(warm_start=True, x=mat_X_ws, s=s_ws)
    # sol = solver.solve(warm_start=True, x=mat_X_ws)
    # sol = solver.solve()

    # print(sol)
    # exit(0)
    out = dict(
        sdp_objval=-sol['info']['pobj'],
        sdp_solvetime=sol['info']['solve_time'] / 1000,
        num_cones=len(PSD_cones),
        x_dim=x_dim,
        num_Aeq=len(Aeq),
        # Aeq_nnz=Aeq.count_nonzero(),
        num_Au=len(Au),
        # Au_nnz=Au.count_nonzero(),
        num_Al=len(Al),
        # Al_nnz=Al.count_nonzero(),
        A_nnz = A.count_nonzero(),
    )

    # return -sol['info']['pobj'], sol['info']['solve_time'] / 1000
    return out
