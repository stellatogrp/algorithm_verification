import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.linear_step import LinearStep
from algocert.init_set.box_set import BoxSet
from algocert.init_set.l2_ball_set import L2BallSet
from algocert.init_set.stack_set import StackSet
from algocert.init_set.zero_set import ZeroSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


class NUM(object):

    def __init__(self, orig_m, orig_n, c_c, c_r=.05, seed=0):
        self.seed = seed
        self.orig_m = orig_m
        self.orig_n = orig_n
        self.c_c = c_c
        self.c_r = c_r
        self._generate_NUM_data()

    def _generate_NUM_data(self):
        m, n = self.orig_m, self.orig_n
        np.random.seed(self.seed)
        R = np.random.binomial(n=1, p=0.4, size=(m, n))
        # R = np.random.binomial(n=1, p=1, size=(m, n))
        # print(R)
        # exit(0)
        self.R = R
        A = np.vstack([R, -np.eye(n), np.eye(n)])
        M = spa.bmat([
            [None, A.T],
            [-A, None]
        ]).toarray()
        self.A = A
        self.M = M
        w = -np.random.uniform(0, 1, size=n)
        t = 1 * np.ones(n)
        self.w = w
        self.t = t

    def generate_CP(self, K):
        m, n = self.A.shape
        obj_w = self.w
        M = self.M
        # print(M.shape)
        # print(m + n)
        k = m + n
        MpI = spa.csc_matrix(M) + spa.eye(k)
        # print(MpI.toarray())
        MpIinv = spa.linalg.inv(MpI)

        Ik = spa.eye(k)

        z = Iterate(k, name='z')
        u = Iterate(k, name='u')
        w = Iterate(k, name='w')
        u_tilde = Iterate(k, name='u_tilde')
        q = Parameter(k, name='q')

        # np.hstack([w, c_l, np.zeros(n), t])
        # q_stack = [obj_w, c, np.zeros(self.orig_n), self.t]
        # q_set = StackSet(q, q_stack)
        # [w, c_l, np.zeros(n), t]
        c_l = 0.9 * np.ones(self.orig_m)
        c_u = np.ones(self.orig_m)
        q_l = np.hstack([obj_w, c_l, np.zeros(n), self.t])
        q_u = np.hstack([obj_w, c_u, np.zeros(n), self.t])
        q_l = q_l.reshape(-1, 1)
        q_u = q_u.reshape(-1, 1)

        q_set = BoxSet(q, q_l, q_u)

        s1_D = MpI
        s1_A = spa.bmat([[Ik, -Ik]])
        s1_b = np.zeros((k, 1))

        step1 = LinearStep(u, [z, q], D=s1_D, A=s1_A, b=s1_b, Dinv=MpIinv)

        # step 2
        s2_D = Ik
        s2_A = spa.bmat([[2 * Ik, -Ik]])
        s2_b = np.zeros((k, 1))

        step2 = LinearStep(w, [u, z], D=s2_D, A=s2_A, b=s2_b, Dinv=s2_D)

        # step 3
        nonneg_ranges = (n, m + n)
        # nonneg_ranges = (m + n - 1, m + n)
        # step3 = PartialNonNegProjStep(u_tilde, w, nonneg_ranges)
        step3 = NonNegProjStep(u_tilde, w, nonneg_ranges=nonneg_ranges)
        # step3 = NonNegProjStep(u_tilde, w, nonneg_ranges=None)
        # step3 = LinearStep(u_tilde, [w], D=Ik, A=Ik, b=s2_b, Dinv=Ik)

        # step 4
        s4_D = Ik
        s4_A = spa.bmat([[Ik, Ik, -Ik]])
        s4_b = np.zeros((k, 1))

        step4 = LinearStep(z, [z, u_tilde, u], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)
        # step4 = LinearStep(z, [z, w, u], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)

        steps = [step1, step2, step3, step4]

        z_set = ZeroSet(z)

        init_sets = [z_set]
        param_sets = [q_set]
        obj = ConvergenceResidual(z)

        return CertificationProblem(K, init_sets, param_sets, obj, steps)

    def generate_CP_ball(self, K):
        m, n = self.A.shape
        obj_w = self.w
        M = self.M
        # print(M.shape)
        # print(m + n)
        k = m + n
        MpI = spa.csc_matrix(M) + spa.eye(k)
        # print(MpI.toarray())
        MpIinv = spa.linalg.inv(MpI)

        Ik = spa.eye(k)

        z = Iterate(k, name='z')
        u = Iterate(k, name='u')
        w = Iterate(k, name='w')
        u_tilde = Iterate(k, name='u_tilde')
        c = Parameter(self.orig_m, name='c')
        c_set = L2BallSet(c, self.c_c, self.c_r)
        # print(self.orig_m + 3 * self.orig_n)
        q = Parameter(k, name='q')

        # np.hstack([w, c_l, np.zeros(n), t])
        q_stack = [(obj_w, obj_w), c, (np.zeros(self.orig_n), np.zeros(self.orig_n)), (self.t, self.t)]
        q_set = StackSet(q, q_stack)

        s1_D = MpI
        s1_A = spa.bmat([[Ik, -Ik]])
        s1_b = np.zeros((k, 1))

        step1 = LinearStep(u, [z, q], D=s1_D, A=s1_A, b=s1_b, Dinv=MpIinv)

        # step 2
        s2_D = Ik
        s2_A = spa.bmat([[2 * Ik, -Ik]])
        s2_b = np.zeros((k, 1))

        step2 = LinearStep(w, [u, z], D=s2_D, A=s2_A, b=s2_b, Dinv=s2_D)

        # step 3
        nonneg_ranges = (n, m + n)
        # nonneg_ranges = (m + n - 1, m + n)
        # step3 = PartialNonNegProjStep(u_tilde, w, nonneg_ranges)
        step3 = NonNegProjStep(u_tilde, w, nonneg_ranges=nonneg_ranges)
        # step3 = NonNegProjStep(u_tilde, w, nonneg_ranges=None)
        # step3 = LinearStep(u_tilde, [w], D=Ik, A=Ik, b=s2_b, Dinv=Ik)

        # step 4
        s4_D = Ik
        s4_A = spa.bmat([[Ik, Ik, -Ik]])
        s4_b = np.zeros((k, 1))

        step4 = LinearStep(z, [z, u_tilde, u], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)
        # step4 = LinearStep(z, [z, w, u], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)

        steps = [step1, step2, step3, step4]

        z_set = ZeroSet(z)

        init_sets = [z_set]
        param_sets = [c_set, q_set]
        obj = ConvergenceResidual(z)

        return CertificationProblem(K, init_sets, param_sets, obj, steps)

    def test_cp_prob(self):
        w = self.w
        t = self.t
        R = self.R
        c = self.c_c - self.c_r

        f = cp.Variable(R.shape[1])
        # print('w:', w)
        obj = cp.Minimize(w @ f)
        constraints = [R @ f <= c.reshape(-1, ), f >= 0, f <= t]
        problem = cp.Problem(obj, constraints)
        problem.solve()
        print(np.round(f.value, 4))

def main():
    m = 2
    n = 3
    K = 1
    c_c = np.ones((m, 1))
    instance = NUM(m, n, c_c, seed=1)
    print(instance.R)
    print(instance.M.shape)
    instance.test_cp_prob()
    # instance.generate_CP(1)

    CP = instance.generate_CP_ball(K)
    sdp_g, sdp_gtime = CP.solve(solver_type='GLOBAL', add_bounds=True)

    CP2 = instance.generate_CP_ball(K)
    out_sdp = CP2.solve(solver_type='SDP_CUSTOM')
    print(out_sdp)

    print(sdp_g, out_sdp['sdp_objval'])


if __name__ == '__main__':
    main()
