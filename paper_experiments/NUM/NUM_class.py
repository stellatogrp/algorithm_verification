import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.linear_max_proj_step import LinearMaxProjStep
from algocert.high_level_alg_steps.linear_step import LinearStep
from algocert.init_set.box_set import BoxSet
from algocert.init_set.l2_ball_set import L2BallSet
from algocert.init_set.stack_set import StackSet
from algocert.init_set.zero_set import ZeroSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


class NUM(object):

    def __init__(self, orig_m, orig_n, c_c, c_r=.1, seed=0, c_seed=1):
        self.seed = seed
        self.c_seed = c_seed
        self.orig_m = orig_m
        self.orig_n = orig_n
        self.c_c = c_c
        self.c_r = c_r
        self._generate_NUM_data()

    def _generate_NUM_data(self):
        m, n = self.orig_m, self.orig_n
        np.random.seed(self.seed)
        R = np.random.binomial(n=1, p=0.6, size=(m, n))
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
        t = 2 * np.ones(n)
        self.w = w
        self.t = t
        self.c_samp = self.sample_c()

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
        # w = Iterate(k, name='w')
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

        # # step 2
        # s2_D = Ik
        # s2_A = spa.bmat([[2 * Ik, -Ik]])
        # s2_b = np.zeros((k, 1))

        # step2 = LinearStep(w, [u, z], D=s2_D, A=s2_A, b=s2_b, Dinv=s2_D)

        # # step 3
        # nonneg_ranges = (n, m + n)
        # # nonneg_ranges = (m + n - 1, m + n)
        # # step3 = PartialNonNegProjStep(u_tilde, w, nonneg_ranges)
        # step3 = NonNegProjStep(u_tilde, w, nonneg_ranges=nonneg_ranges)
        # # step3 = NonNegProjStep(u_tilde, w, nonneg_ranges=None)
        # # step3 = LinearStep(u_tilde, [w], D=Ik, A=Ik, b=s2_b, Dinv=Ik)

        # step 2
        s2_A = spa.bmat([[2 * Ik, -Ik]])
        s2_b = np.zeros((k, 1))

        step2 = LinearMaxProjStep(u_tilde, [u, z], A=s2_A, b=s2_b)

        # step 4
        s4_D = Ik
        s4_A = spa.bmat([[Ik, Ik, -Ik]])
        s4_b = np.zeros((k, 1))

        step4 = LinearStep(z, [z, u_tilde, u], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)
        # step4 = LinearStep(z, [z, w, u], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)

        # steps = [step1, step2, step3, step4]
        steps = [step1, step2, step4]

        z_set = ZeroSet(z)

        init_sets = [z_set]
        param_sets = [q_set]
        obj = ConvergenceResidual(z)

        return CertificationProblem(K, init_sets, param_sets, obj, steps)

    def generate_CP_ball(self, K, init_type='cs'):
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
        # w = Iterate(k, name='w')
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

        # # step 2
        # s2_D = Ik
        # s2_A = spa.bmat([[2 * Ik, -Ik]])
        # s2_b = np.zeros((k, 1))

        # step2 = LinearStep(w, [u, z], D=s2_D, A=s2_A, b=s2_b, Dinv=s2_D)

        # # step 3
        # nonneg_ranges = (n, m + n)
        # # nonneg_ranges = (m + n - 1, m + n)
        # # step3 = PartialNonNegProjStep(u_tilde, w, nonneg_ranges)
        # step3 = NonNegProjStep(u_tilde, w, nonneg_ranges=nonneg_ranges)
        # # step3 = NonNegProjStep(u_tilde, w, nonneg_ranges=None)
        # # step3 = LinearStep(u_tilde, [w], D=Ik, A=Ik, b=s2_b, Dinv=Ik)

        # step 2
        s2_A = spa.bmat([[2 * Ik, -Ik]])
        s2_b = np.zeros((k, 1))

        step2 = LinearMaxProjStep(u_tilde, [u, z], A=s2_A, b=s2_b, proj_ranges=(n, m + n))

        # step 4
        s4_D = Ik
        s4_A = spa.bmat([[Ik, Ik, -Ik]])
        s4_b = np.zeros((k, 1))

        step4 = LinearStep(z, [z, u_tilde, u], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)
        # step4 = LinearStep(z, [z, w, u], D=s4_D, A=s4_A, b=s4_b, Dinv=s4_D)

        # steps = [step1, step2, step3, step4]
        steps = [step1, step2, step4]

        if init_type == 'ws':
            ws_sol = self.test_cp_prob()
            z_set = L2BallSet(z, ws_sol, 0)
        elif init_type == 'heur':
            heur = self.heuristic_start()
            z_set = L2BallSet(z, heur, 0)
        else:
            z_set = ZeroSet(z)

        init_sets = [z_set]
        param_sets = [c_set, q_set]
        obj = ConvergenceResidual(z)

        return CertificationProblem(K, init_sets, param_sets, obj, steps)

    def test_cp_prob(self):
        w = self.w
        c = self.c_samp

        # f = cp.Variable(R.shape[1])
        # # print('w:', w)
        # obj = cp.Minimize(w @ f)
        # constraints = [R @ f <= c.reshape(-1, ), f >= 0, f <= t]
        # problem = cp.Problem(obj, constraints)
        # problem.solve()
        # print(np.round(f.value, 4))

        A = self.A
        rhs = np.hstack([c.reshape(-1, ), np.zeros(self.orig_n), self.t])
        f = cp.Variable(A.shape[1])
        # print('w:', w)
        obj = cp.Minimize(w @ f)
        constraints = [A @ f <= rhs]
        problem = cp.Problem(obj, constraints)
        problem.solve()
        print(np.round(f.value, 4))
        print('dual var:', np.round(constraints[0].dual_value, 4))

        pd_sol = np.hstack([f.value, constraints[0].dual_value])
        # print(np.round(pd_sol, 4))
        return np.round(pd_sol, 4).reshape(-1, 1)

    def heuristic_start(self):
        c_samp = self.sample_c()
        k = np.sum(self.R, axis=1)

        alphas = c_samp.reshape(-1, ) / k
        # print(alphas)
        alpha_min = np.min(alphas)
        # print(alpha_min)

        m, n = self.A.shape

        out = np.zeros(m + n)
        out[:n] = alpha_min
        # print(out)

        # exit(0)
        return out.reshape(-1, 1)

    def sample_c(self):
        np.random.seed(self.c_seed)
        c = self.c_c
        r = self.c_r
        sample = np.random.normal(0, 1, c.shape[0])
        sample = np.random.uniform(0, r) * sample / np.linalg.norm(sample)
        # print(np.linalg.norm(sample))
        # print(sample.reshape(-1, 1) + c)
        return sample.reshape(-1, 1) + c


class NUM_simple(object):

    def __init__(self, orig_m, orig_n, c_c, c_r=.1, seed=0, c_seed=1):
        self.seed = seed
        self.c_seed = c_seed
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
        A = np.vstack([R, -np.eye(n)])
        M = spa.bmat([
            [None, A.T],
            [-A, None]
        ]).toarray()
        self.A = A
        self.M = M
        w = -np.random.uniform(0, 1, size=n)
        self.w = w
        self.c_samp = self.sample_c()


    def generate_CP_ball(self, K, warm_start=False):
        self.test_cp_prob()
        exit(0)


    def test_cp_prob(self):
        w = self.w
        c = self.c_samp

        A = self.A
        rhs = np.hstack([c.reshape(-1, ), np.zeros(self.orig_n)])
        f = cp.Variable(A.shape[1])
        # print('w:', w)
        obj = cp.Minimize(w @ f)
        constraints = [A @ f <= rhs]
        problem = cp.Problem(obj, constraints)
        problem.solve()
        print(np.round(f.value, 4))
        exit(0)
        # print('dual var:', np.round(constraints[0].dual_value, 4))

        pd_sol = np.hstack([f.value, constraints[0].dual_value])
        # print(np.round(pd_sol, 4))
        return np.round(pd_sol, 4).reshape(-1, 1)


    def sample_c(self):
        np.random.seed(self.c_seed)
        c = self.c_c
        r = self.c_r
        sample = np.random.normal(0, 1, c.shape[0])
        sample = np.random.uniform(0, r) * sample / np.linalg.norm(sample)
        # print(np.linalg.norm(sample))
        # print(sample.reshape(-1, 1) + c)
        return sample.reshape(-1, 1) + c

def main():
    m = 3
    n = 5
    K = 5
    c_r = .1
    ws = True
    c_c = np.ones((m, 1))
    instance = NUM(m, n, c_c, c_r=c_r, seed=3)
    # print(instance.R)
    # exit(0)
    # print(instance.M.shape)
    test_pd = instance.test_cp_prob()
    print(test_pd)
    # instance.generate_CP(1)

    CP = instance.generate_CP_ball(K, warm_start=ws)
    sdp_g, sdp_gtime = CP.solve(solver_type='GLOBAL', add_bounds=True)

    CP2 = instance.generate_CP_ball(K, warm_start=ws)
    out_sdp = CP2.solve(solver_type='SDP_CUSTOM')
    print(out_sdp)

    print(instance.R)
    print(sdp_g, out_sdp['sdp_objval'])

    # instance.sample_c()


if __name__ == '__main__':
    main()
