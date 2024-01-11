import cvxpy as cp
import numpy as np
import scipy.sparse as spa

from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.linear_max_proj_step import LinearMaxProjStep
from algocert.high_level_alg_steps.linear_step import LinearStep
from algocert.init_set.l2_ball_set import L2BallSet
from algocert.init_set.zero_set import ZeroSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


class ISTA(object):

    def __init__(self, m, n, b_c, b_r, lambd=0.01, seed=0):
        self.seed = seed
        self.c_seed = 0
        self.m = m
        self.n = n
        self.b_c = b_c
        self.b_r = b_r
        self.lambd = lambd
        self._generate_A_mat()

    def _generate_A_mat(self):
        np.random.seed(self.seed)
        self.A = np.random.randn(self.m, self.n)

    def get_t_opt(self):
        A = self.A
        ATA = A.T @ A
        eigs = np.linalg.eigvals(ATA)
        mu = np.min(eigs)
        L = np.max(eigs)
        # print(2/(mu + L))
        return np.real(2 / (mu + L))

    def generate_CP(self, K, t=None):
        m, n = self.m, self.n
        A = spa.csc_matrix(self.A)
        ATA = A.T @ A
        In = spa.eye(n)

        # print((In - t * ATA).shape, (t * A.T).shape)
        if t is None:
            t = self.get_t_opt()
        C = spa.bmat([[In - t * ATA, t * A.T]])
        D = spa.eye(n, n)
        b_const = np.zeros((n, 1))
        lambd_ones = self.lambd * t * np.ones((n, 1))

        y = Iterate(n, name='y')
        x = Iterate(n, name='x')
        u = Iterate(n, name='w')
        v = Iterate(n, name='u')
        b = Parameter(m, name='b')

        xset = ZeroSet(x)
        bset = L2BallSet(b, self.b_c, self.b_r)

        step1 = LinearStep(y, [x, b], D=D, A=C, b=b_const, Dinv=D)
        # NonNegProjStep(x, y)
        step2 = LinearMaxProjStep(u, [y], A=np.eye(n), b=-lambd_ones)
        step3 = LinearMaxProjStep(v, [y], A=-np.eye(n), b=-lambd_ones)

        s4A = spa.bmat([[In, -In]])
        step4 = LinearStep(x, [u, v], D=D, A=s4A, b=b_const, Dinv=D)

        # step1 = LinearMaxProjStep(x, [x, b], A=C, b=b_const)
        steps = [step1, step2, step3, step4]

        var_sets = [xset]
        param_sets = [bset]
        obj = ConvergenceResidual(x)

        return CertificationProblem(K, var_sets, param_sets, obj, steps)

    def generate_betas(self, K):
        beta = 1
        beta_list = [beta]
        for _ in range(K):
            beta_new = .5 * (1 + np.sqrt(1 + 4 * beta_list[-1] ** 2))
            beta_list.append(beta_new)
        scalar_beta_list = []
        # print(beta_list)
        for i in range(K):
            scalar_beta_list.append((beta_list[i] - 1) / beta_list[i + 1])
        # print(scalar_beta_list)
        return scalar_beta_list

    def generate_FISTA_CP(self, K, t=None):
        m, n = self.m, self.n
        A = spa.csc_matrix(self.A)
        ATA = A.T @ A
        In = spa.eye(n)

        # print((In - t * ATA).shape, (t * A.T).shape)
        if t is None:
            t = self.get_t_opt()
        C = spa.bmat([[In - t * ATA, t * A.T]])
        D = spa.eye(n, n)
        b_const = np.zeros((n, 1))
        lambd_ones = self.lambd * t * np.ones((n, 1))
        beta_list = self.generate_betas(K)

        y = Iterate(n, name='y')
        u = Iterate(n, name='u')
        v = Iterate(n, name='v')
        w = Iterate(n, name='w')
        z = Iterate(n, name='z')
        b = Parameter(m, name='b')

        zset = ZeroSet(z)
        wset = ZeroSet(w)
        bset = L2BallSet(b, self.b_c, self.b_r)

        step1 = LinearStep(y, [w, b], D=D, A=C, b=b_const, Dinv=D)
        # NonNegProjStep(x, y)
        step2 = LinearMaxProjStep(u, [y], A=np.eye(n), b=-lambd_ones)
        step3 = LinearMaxProjStep(v, [y], A=-np.eye(n), b=-lambd_ones)

        s4A = spa.bmat([[In, -In]])
        step4 = LinearStep(z, [u, v], D=D, A=s4A, b=b_const, Dinv=D)

        s5C = []
        for beta in beta_list:
            # print(beta)
            s5C.append(spa.bmat([[(1 + beta) * spa.eye(n), - beta * spa.eye(n)]]))
            # s5C.append(spa.bmat([[beta * spa.eye(n)]]))
        step5 = LinearStep(w, [z, z], D=D, A=s5C, b=b_const, Dinv=D)
        # step5 = LinearStep(w, [z], D=D, A=s5C, b=b_const, Dinv=D)

        # step1 = LinearMaxProjStep(x, [x, b], A=C, b=b_const)
        steps = [step1, step2, step3, step4, step5]

        var_sets = [zset, wset]
        param_sets = [bset]
        obj = ConvergenceResidual(z)

        return CertificationProblem(K, var_sets, param_sets, obj, steps)

    def test_cp_prob(self):
        A = self.A
        b = self.sample_c().reshape(-1, )
        x = cp.Variable(A.shape[1])
        obj = cp.Minimize(.5 * cp.sum_squares(A @ x - b) + self.lambd * cp.norm(x, 1))
        prob = cp.Problem(obj)
        prob.solve()
        print(np.round(x.value, 4))

        return np.round(x.value, 4)

    def sample_c(self):
        # np.random.seed(self.c_seed)
        c = self.b_c
        r = self.b_r
        sample = np.random.normal(0, 1, c.shape[0])
        sample = np.random.uniform(0, r) * sample / np.linalg.norm(sample)
        # print(np.linalg.norm(sample))
        # print(sample.reshape(-1, 1) + c)
        return sample.reshape(-1, 1) + c
