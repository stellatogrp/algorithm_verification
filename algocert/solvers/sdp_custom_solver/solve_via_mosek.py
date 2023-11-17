import numpy as np
import scipy.sparse as spa
from mosek.fusion import Domain, Expr, Matrix, Model, ObjectiveSense


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


def spa_to_mosek_mat(M):
    i, j, vals = spa.find(M)
    return Matrix.sparse(M.shape[0], M.shape[1], i, j, vals)


def solve_via_mosek(C, A_vals, b_lvals, b_uvals, PSD_cones, problem_dim):
    print('----solving via mosek directly----')
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


    with Model() as M:
        import sys
        M.setLogHandler(sys.stdout)
        x = M.variable(x_dim, Domain.unbounded())
        # x = M.variable(x_dim, Domain.inSVecPSDCone(x_dim))
        M.objective(ObjectiveSense.Minimize, Expr.dot(c, x))

        for cone in PSD_cones:
            H = cone.get_sparse_Hsymm_mat(problem_dim)
            z_dim = H.shape[0]
            H = spa_to_mosek_mat(H)
            z = M.variable(z_dim, Domain.inSVecPSDCone(z_dim))
            M.constraint(Expr.sub(Expr.mul(H, x), z), Domain.equalsTo(0))

        for Ai, bli, bui in zip(A_vals, b_lvals, b_uvals):
            # if bli == bui:
            #     M.constraint(Expr.dot(Ai_vec.todense(), x), Domain.equalsTo(bli))
            if bli == bui:
                Aeq.append(spa_vec(Ai))
                beq.append(bli)
            else:
                if bui < np.inf:
                    Au.append(spa_vec(Ai))
                    bu.append(bui)
                if bli > -np.inf:
                    Al.append(spa_vec(Ai))
                    bl.append(bli)
        Aeq = spa.vstack(Aeq)
        # i, j, vals = spa.find(Aeq)
        # Aeq_mat = Matrix.sparse(Aeq.shape[0], Aeq.shape[1], i, j, vals)
        Aeq_mat = spa_to_mosek_mat(Aeq)
        # beq = np.array(beq)
        # print(Aeq.shape, beq.shape)
        M.constraint(Expr.mul(Aeq_mat, x), Domain.equalsTo(beq))

        Au = spa.vstack(Au)
        # i, j, vals = spa.find(Au)
        # Au_mat = Matrix.sparse(Au.shape[0], Au.shape[1], i, j, vals)
        Au_mat = spa_to_mosek_mat(Au)
        M.constraint(Expr.mul(Au_mat, x), Domain.lessThan(bu))

        if len(Al) > 0:
            Al = spa.vstack(Al)
            # i, j, vals = spa.find(Al)
            # Al_mat = Matrix.sparse(Al.shape[0], Al.shape[1], i, j, vals)
            Al_mat = spa_to_mosek_mat(Al)
            M.constraint(Expr.mul(Al_mat, x), Domain.greaterThan(bl))
        else:
            Al = spa.csc_matrix(np.array([]))

        tol = 1e-5
        M.setSolverParam('intpntCoTolDfeas', tol)
        M.setSolverParam('intpntCoTolPfeas', tol)
        M.setSolverParam('intpntCoTolRelGap', tol)

        M.solve()
        # M->setSolverParam("intpntMaxIterations", 400)
        # res = x.level()
        # print(M.getPrimalSolutionStatus())
        # print(M.primalObjValue())
        # print(M.getSolverDoubleInfo('optimizerTime'))
        solvetime = M.getSolverDoubleInfo('optimizerTime')
        objval = -M.primalObjValue()

    out = dict(
        sdp_objval=objval,
        sdp_solvetime=solvetime,
        num_cones=len(PSD_cones),
        problem_dim=problem_dim,
        x_dim=x_dim,
        num_Aeq=Aeq.shape[0],
        Aeq_nnz=Aeq.count_nonzero(),
        num_Au=Au.shape[0],
        Au_nnz=Au.count_nonzero(),
        num_Al=Al.shape[0],
        Al_nnz=Al.count_nonzero(),
    )
    return out
