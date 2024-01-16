import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from algocert.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.linear_max_proj_step import LinearMaxProjStep
from algocert.high_level_alg_steps.linear_step import LinearStep
from algocert.init_set.box_set import BoxSet
from algocert.init_set.stack_set import StackSet
from algocert.init_set.zero_set import ZeroSet
from algocert.objectives.block_convergence_residual import BlockConvergenceResidual
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


class Car2D(object):

    def __init__(self, h=0.1, eta=0.1, mass=1, rho=2, vmax=2, xmax=2, smax=.2, T=2):
        self.nx = 4
        self.nv = 2
        self.h = h
        self.eta = eta
        self.mass = mass
        self.rho = rho
        self.vmax = vmax
        self.xmax = xmax
        self.smax = smax
        self.T = T
        self.nx = 4
        self.nv = 2
        self.A = np.array([
            [1, 0, h, 0],
            [0, 1, 0, h],
            [0, 0, 1-h*eta/mass, 0],
            [0, 0, 0, 1-h*eta/mass]
        ])
        self.B = np.array([
            [0, 0],
            [0, 0],
            [h/mass, 0],
            [0, h/mass]
        ])
        self.C = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        self._form_block_matrices()

    def _form_block_matrices(self):
        # print('forming')
        nx, nv = self.nx, self.nv
        C = self.C
        CTC = C.T @ C
        rho = self.rho
        T = self.T
        R = rho / 2 * spa.eye(nv)
        H = spa.block_diag([spa.kron(spa.eye(T), CTC), spa.kron(spa.eye(T-1), R)])
        # print(H.todense(), H.shape)
        Ax = spa.kron(spa.eye(T), -spa.eye(nx)) + spa.kron(spa.eye(T, k=-1), self.A)
        Ax[:nx, :nx] = spa.eye(nx)
        # print(Ax.todense(), Ax.shape)
        Bv = spa.kron(spa.vstack([spa.csc_matrix((1, T-1)), spa.eye(T-1)]), self.B)
        # print(Bv.todense(), Bv.shape)

        Aeq = spa.hstack([Ax, Bv])
        # print(Aeq.shape)
        # print(Aeq.todense())
        Aineq = spa.eye(T * nx + (T-1) * nv)
        M = spa.vstack([Aeq, Aineq])
        # print(M.shape)

        # diffs = np.zeros(((T-2)* nv, (T-1)* nv))
        # for i in range((T-2) * nv):
        #     diffs[i, i] = -1
        #     diffs[i, i + nv] = 1
        diffs = np.eye((T-1) * nv) - np.eye((T-1) * nv, k=-nv)
        # swap the u1 uinit constraint to the end
        # diffs[[0, -1]] = diffs[[-1, 0]]
        # print(diffs)

        # for i in range(nv):
        #     print(i, -i-1)
        #     diffs[[i, -i-1]] = diffs[[-i-1, i]]

        # diffs = np.flipud(diffs)
        # print(diffs)
        # exit(0)

        self.H = H
        self.M = M
        self.diffs = spa.csc_matrix(diffs)
        # print(Aeq.shape, M.shape)

    def test_with_cvxpy(self):
        print('----full cvxpy with all states----')
        H = self.H
        M = self.M
        nx, nv = self.nx, self.nv
        T = self.T
        n = T * nx + (T - 1) * nv

        vmax = self.vmax
        # xmax = self.xmax
        xmax = 10
        smax = self.smax

        xinit = np.array([1, 1, 0, 0])
        uinit = np.array([0, 0])
        # uinit = np.array([-.5, -.5])
        l = np.zeros(M.shape[0])
        u = np.zeros(M.shape[0])
        l[:nx] = xinit
        u[:nx] = xinit

        # l = np.hstack([l, np.kron(np.ones(T), xmax * np.ones(nx)), np.kron(np.ones(T-1), vmax * np.ones(nv))])
        l[-n:] = np.hstack([np.kron(np.ones(T), -xmax * np.ones(nx)), np.kron(np.ones(T-1), -vmax * np.ones(nv))])
        u[-n:] = np.hstack([np.kron(np.ones(T), xmax * np.ones(nx)), np.kron(np.ones(T-1), vmax * np.ones(nv))])

        s = smax * np.ones((T-2) * nv)
        print(M.shape)
        print(self.diffs.shape)
        M = spa.vstack([M, spa.hstack([spa.csc_matrix(((T-1) * nv, T*nx)), self.diffs])])
        # print(s.shape)

        l = np.hstack([l, -smax + uinit, -s])
        u = np.hstack([u, smax + uinit, s])
        # l = np.hstack([l, -s, -smax + uinit])
        # u = np.hstack([u, s, smax + uinit])

        # print(l)
        # print(u)
        # exit(0)

        # print(M.shape)
        # print(l, u, l.shape, u.shape)

        x = cp.Variable(n)
        obj = cp.Minimize(.5 * cp.quad_form(x, H))
        constraints = [l <= M @ x, M @ x <= u]
        prob = cp.Problem(obj, constraints)
        res = prob.solve()
        print('res:', res)
        print('xsol:')
        print(np.round(x.value, 4))

    def test_simplified_cvxpy(self):
        print('----cvxpy with only initial state----')
        nx, nv = self.nx, self.nv
        T = self.T
        rho = self.rho
        vmax = self.vmax
        vmin = -vmax
        xinit = np.array([1, 1, 0, 0])
        uinit = np.array([0, 0])
        # uinit = np.array([-.5, -.5])
        A, B, C = self.A, self.B, self.C
        P = spa.kron(spa.eye(T), C.T @ C)
        R = spa.kron(spa.eye(T-1), rho / 2 * spa.eye(nv))

        # form SX
        SX = []
        for i in range(T):
            SX.append(np.linalg.matrix_power(A, i))
        SX = np.vstack(SX)
        print('SX shape:', SX.shape)
        # print(SX)

        SV = spa.csc_matrix(((T-1) * nx, (T-1) * nv))
        for i in range(T-1):
            # print(np.eye(T-1, k=-i))
            AiB = np.linalg.matrix_power(A, i) @ B
            SV += spa.kron(spa.eye(T-1, k=-i), AiB)
        # SV += spa.kron(B, spa.eye(T-1))
        # print(np.kron(np.eye(T-1, k=0), B))
        SV = spa.vstack([spa.csc_matrix((nx, (T-1) * nv)), SV])
        print(SV.shape)
        # print(SV.todense())
        diffs = self.diffs
        smax = self.smax
        # print(diffs.todense())

        s = smax * np.ones((T-2) * nv)
        smin_vec = np.hstack([-smax + uinit, -s])
        smax_vec = np.hstack([smax + uinit, s])
        # smin_vec = np.hstack([-s, -smax + uinit])
        # smax_vec = np.hstack([s, smax + uinit])

        x1 = cp.Variable(nx)
        X = cp.Variable(T * nx)
        V1 = cp.Variable((T-1) * nv)
        obj = cp.Minimize(.5 * cp.quad_form(X, P) + .5 * cp.quad_form(V1, R))
        constraints = [
            X == SX @ x1 + SV @ V1,
            x1 == xinit,
            vmin <= V1, V1 <= vmax,
            smin_vec <= diffs @ V1, diffs @ V1 <= smax_vec,
        ]
        prob = cp.Problem(obj, constraints)
        res = prob.solve()
        print(res)
        print('xinit sol:')
        print(np.round(x1.value, 4))
        print('V sol:')
        print(np.round(V1.value, 4))
        # exit(0)

        print('----cvxpy with all things simplified out----')
        X = cp.Variable(nx + (T-1) * nv)
        # print((SX.T @ P @ SX).shape)
        # print((SX.T @ P @ SV).shape)
        # print((SV.T @ P @ SV + R).shape)

        H = spa.bmat([
            [SX.T @ P @ SX, SX.T @ P @ SV],
            [SV.T @ P @ SX, SV.T @ P @ SV + R]
        ])

        l = np.hstack([xinit, np.kron(np.ones(T-1), vmin * np.ones(nv))])
        u = np.hstack([xinit, np.kron(np.ones(T-1), vmax * np.ones(nv))])

        M = spa.eye(nx + (T-1) * nv)
        s = self.smax * np.ones((T-2) * nv)

        print(M.shape)
        print(self.diffs.shape)
        # print(spa.hstack([spa.csc_matrix(((T-2) * nv, nx)), self.diffs]))
        M = spa.vstack([M, spa.hstack([spa.csc_matrix(((T-1) * nv, nx)), self.diffs])])
        # u = np.hstack([u, s])
        # l = np.hstack([l, -s])
        l = np.hstack([l, -smax + uinit, -s])
        u = np.hstack([u, smax + uinit, s])
        # l = np.hstack([l, -s, -smax + uinit])
        # u = np.hstack([u, s, smax + uinit])
        # exit(0)

        obj = cp.Minimize(.5 * cp.quad_form(X, H))
        constraints = [l <= M @ X, M @ X <= u]
        prob = cp.Problem(obj, constraints)
        res = prob.solve()
        print(res)
        print('sol:')
        print(np.round(X.value, 4))

        print(H.shape, M.shape)

    def get_QP_data(self):
        nx, nv = self.nx, self.nv
        T = self.T
        rho = self.rho
        vmax = self.vmax
        vmin = -vmax
        np.array([1, 1, 0, 0])
        # if uinit is None:
            # uinit = np.array([0, 0])
        # uinit = np.array([-.5, -.5])
        A, B, C = self.A, self.B, self.C
        P = spa.kron(spa.eye(T), C.T @ C)
        R = spa.kron(spa.eye(T-1), rho / 2 * spa.eye(nv))

        # form SX
        SX = []
        for i in range(T):
            SX.append(np.linalg.matrix_power(A, i))
        SX = np.vstack(SX)
        # print('SX shape:', SX.shape)
        # print(SX)

        SV = spa.csc_matrix(((T-1) * nx, (T-1) * nv))
        for i in range(T-1):
            AiB = np.linalg.matrix_power(A, i) @ B
            SV += spa.kron(spa.eye(T-1, k=-i), AiB)
        SV = spa.vstack([spa.csc_matrix((nx, (T-1) * nv)), SV])
        # print(SV.shape)
        smax = self.smax

        s = smax * np.ones((T-2) * nv)

        H = spa.bmat([
            [SX.T @ P @ SX, SX.T @ P @ SV],
            [SV.T @ P @ SX, SV.T @ P @ SV + R]
        ])

        M = spa.eye(nx + (T-1) * nv)
        s = self.smax * np.ones((T-2) * nv)
        M = spa.vstack([M, spa.hstack([spa.csc_matrix(((T-1) * nv, nx)), self.diffs])])

        l1 = np.kron(np.ones(T-1), vmin * np.ones(nv))
        l2 = -s
        u1 = np.kron(np.ones(T-1), vmax * np.ones(nv))
        u2 = s

        # l = np.hstack([np.kron(np.ones(T-1), vmin * np.ones(nv)), -smax + uinit, -s])
        # u = np.hstack([np.kron(np.ones(T-1), vmax * np.ones(nv)), smax + uinit, s])

        # print(H.shape, M.shape)
        # print(H.todense(), M.todense())
        # print(l.shape, u.shape)
        return H, M, l1, l2, u1, u2

    def solve_via_cvxpy(self, xinit, uinit=None):
        if uinit is None:
            uinit = np.array([0, 0])
        H, M, l1, l2, u1, u2 = self.get_QP_data()

        # l = np.hstack([xinit, l])
        # u = np.hstack([xinit, u])

        l = np.hstack([xinit, l1, -self.smax + uinit, l2])
        u = np.hstack([xinit, u1, self.smax + uinit, u2])

        X = cp.Variable(H.shape[0])
        obj = cp.Minimize(.5 * cp.quad_form(X, H))
        constraints = [l <= M @ X, M @ X <= u]
        prob = cp.Problem(obj, constraints)
        prob.solve()
        # print('opt obj:', res)

        sol = np.round(X.value, 4)
        # print('sol:', sol)
        return sol

    def get_CP(self, K, xinit_min, xinit_max, uinit_min, uinit_max,
               rho_const=True, x0_min=None, x0_max=None):
        P, A, l1, l2, u1, u2 = self.get_QP_data()

        print(P.shape, A.shape)

        m, n = A.shape
        if rho_const:
            rho = np.eye(m)
        else:
            # eq_idx = np.where(np.abs(u - l) <= 1e-5)
            # print(u-l)
            eq_idx = list(range(self.nx))
            print(eq_idx)
            rho = np.ones(m)
            rho[eq_idx] *= 10
            rho = np.diag(rho)
        rho = spa.csc_matrix(rho)
        # rho_inv = np.linalg.inv(rho)
        rho_inv = spa.linalg.inv(rho)
        sigma = 1e-6

        xinit = Parameter(self.nx, name='x_init')
        l = Parameter(m, name='l')
        u = Parameter(m, name='u')

        # print(-self.smax + uinit_min, -self.smax + uinit_max)
        # print(self.smax + uinit_min, self.smax + uinit_max)

        l_min = np.hstack([l1, -self.smax + uinit_min, l2])
        l_max = np.hstack([l1, -self.smax + uinit_max, l2])
        # print(l_max - l_min)

        u_min = np.hstack([u1, self.smax + uinit_min, u2])
        u_max = np.hstack([u1, self.smax + uinit_max, u2])
        # print(u_max - u_min)

        lset_rest = (l_min, l_max)
        uset_rest = (u_min, u_max)

        # paramsets
        xinit_set = BoxSet(xinit, xinit_min.reshape(-1, 1), xinit_max.reshape(-1, 1))
        lset = StackSet(l, [xinit, lset_rest])
        uset = StackSet(u, [xinit, uset_rest])

        paramsets = [xinit_set, lset, uset]

        x = Iterate(n, name='x')
        w = Iterate(m, name='w')
        z = Iterate(m, name='z')
        y = Iterate(m, name='y')

        In = np.eye(n)
        Im = np.eye(m)
        zeros_n = np.zeros((n, 1))
        zeros_m = np.zeros((m, 1))

        # step 1
        # (P + sigma I + A^T rho A) x = sigma x + A^T rho z - A^T y
        # print(P.shape, In.shape, rho.shape, A.shape)
        s1D = spa.csc_matrix(P + sigma * In + A.T @ rho @ A)
        s1A = spa.bmat([[sigma * In, A.T @ rho, -A.T]])
        s1b = zeros_n
        s1Dinv = spa.linalg.inv(s1D)
        step1 = LinearStep(x, [x, z, y], D=s1D, A=s1A, b=s1b, Dinv=s1Dinv)

        # step 2
        s2A = spa.bmat([[A, rho_inv @ Im]])
        s2b = zeros_m
        step2 = LinearMaxProjStep(w, [x, y], A=s2A, b=s2b, l=l)

        # step 3
        step3 = MinWithVecStep(z, w, u=u)

        # step 4
        s4D = spa.eye(m)
        # s5A = spa.bmat([[Im, rho @ A, -rho @ Im]])
        s4A = spa.bmat([[Im, rho @ A, -rho]])
        s4b = zeros_m
        s4Dinv = s4D
        step4 = LinearStep(y, [y, x, z], D=s4D, A=s4A, b=s4b, Dinv=s4Dinv)

        steps = [step1, step2, step3, step4]

        if x0_min is None:
            xset = ZeroSet(x)
        else:
            # xset = L2BallSet(x, ws_x.reshape(-1, 1), 0)
            # x0_min = np.min(shifted_sols, axis=0).reshape((-1, 1))
            # x0_max = np.max(shifted_sols, axis=0).reshape((-1, 1))
            xset = BoxSet(x, x0_min.reshape((-1, 1)), x0_max.reshape((-1, 1)))
        zset = ZeroSet(z)
        yset = ZeroSet(y)

        initsets = [xset, yset, zset]
        obj_block = spa.bmat([[Im, rho_inv]])

        # obj = [ConvergenceResidual(x)]
        # obj = [BlockConvergenceResidual([z, y], obj_block)]
        obj = [ConvergenceResidual(x), BlockConvergenceResidual([z, y], obj_block)]

        return CertificationProblem(K, initsets, paramsets, obj, steps)


def main():
    car = Car2D(T=5)

    # car.test_with_cvxpy()
    # print(car.A)
    # car.test_simplified_cvxpy()
    car.get_QP_data()

    xinit = np.array([1, 1, 0, 0])
    uinit = np.array([-.75, -.75])
    car.solve_via_cvxpy(xinit, uinit=uinit)


if __name__ == '__main__':
    main()
