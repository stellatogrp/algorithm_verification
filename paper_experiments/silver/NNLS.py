import numpy as np
import scipy.sparse as spa

from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.linear_max_proj_step import LinearMaxProjStep
from algocert.high_level_alg_steps.linear_step import LinearStep
from algocert.init_set.l2_ball_set import L2BallSet
from algocert.init_set.zero_set import ZeroSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


class NNLS(object):

    def __init__(self, m, n, b_c, b_r, seed=0):
        self.seed = seed
        self.m = m
        self.n = n
        self.b_c = b_c
        self.b_r = b_r
        self._generate_A_mat()
        self._compute_L_mu()

    def _generate_A_mat(self):
        np.random.seed(self.seed)
        self.A = np.random.randn(self.m, self.n)

    def _compute_L_mu(self):
        A = self.A
        ATA = A.T @ A
        eigs = np.linalg.eigvals(ATA)
        self.L = np.max(eigs)
        self.mu = np.min(eigs)
        self.kappa = self.L / self.mu

    def get_t_opt(self):
        return 2 / (self.mu + self.L)

    def get_silver_steps(self, K):
        L = self.L
        rho = 1 + np.sqrt(2)
        silver_steps = [1 + rho ** ((k & -k).bit_length()-2) for k in range(1, K+1)]
        # print(silver_steps)
        return [alpha / L for alpha in silver_steps]

    def generate_CP(self, t, K):
        m, n = self.m, self.n
        A = spa.csc_matrix(self.A)
        ATA = A.T @ A
        In = spa.eye(n)

        # print((In - t * ATA).shape, (t * A.T).shape)
        if isinstance(t, list):
            C = []
            for t_curr in t:
                C_curr = spa.bmat([[In - t_curr * ATA, t_curr * A.T]])
                C.append(C_curr)
        else:
            C = spa.bmat([[In - t * ATA, t * A.T]])
        D = spa.eye(n, n)
        b_const = np.zeros((n, 1))

        y = Iterate(n, name='y')
        x = Iterate(n, name='x')
        b = Parameter(m, name='b')

        xset = ZeroSet(x)
        bset = L2BallSet(b, self.b_c, self.b_r)

        # step1 = LinearStep(y, [x, b], D=D, A=C, b=b_const, Dinv=D)
        # step2 = NonNegProjStep(x, y)
        # steps = [step1, step2]

        step1 = LinearMaxProjStep(x, [x, b], A=C, b=b_const)
        steps = [step1]

        var_sets = [xset]
        param_sets = [bset]
        obj = ConvergenceResidual(x)

        return CertificationProblem(K, var_sets, param_sets, obj, steps)
