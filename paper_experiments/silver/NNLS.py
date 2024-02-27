import cvxpy as cp
import numpy as np
import scipy.sparse as spa
from scipy.stats import ortho_group

from algoverify.high_level_alg_steps.linear_max_proj_step import LinearMaxProjStep
from algoverify.init_set.l2_ball_set import L2BallSet
from algoverify.init_set.zero_set import ZeroSet
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.variables.iterate import Iterate
from algoverify.variables.parameter import Parameter
from algoverify.verification_problem import VerificationProblem


class NNLS(object):

    def __init__(self, m, n, b_c, b_r, setA=True, ATA_mu=10, ATA_L=100, seed=0):
        self.seed = seed
        self.m = m
        self.n = n
        self.b_c = b_c
        self.b_r = b_r
        if setA:
            self.ATA_mu = ATA_mu
            self.ATA_L = ATA_L
            self._generate_given_eig_A()
        else:
            self._generate_A_mat()
        self._compute_L_mu()

    def _generate_given_eig_A(self):
        np.random.seed(self.seed)
        m, n = self.m, self.n
        mu = self.ATA_mu
        L = self.ATA_L
        print('generating A with given eigvals')
        print(mu, L)

        U = ortho_group.rvs(dim=m)
        U = U[:, :n]
        Sigma = self._generate_sigma(np.sqrt(mu), np.sqrt(L))
        VT = ortho_group.rvs(dim=n)

        self.A = U @ Sigma @ VT

    def _generate_sigma(self, mu, L):
        n = self.n
        out = np.zeros(n)
        out[1:n-1] = np.random.uniform(low=mu, high=L, size=(n-2,))
        out[0] = mu
        out[-1] = L
        return np.diag(out)

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
        spa.eye(n, n)
        b_const = np.zeros((n, 1))

        Iterate(n, name='y')
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

        return VerificationProblem(K, var_sets, param_sets, obj, steps)

    def test_center_cvxpy(self):
        A = self.A
        b = self.b_c.reshape(-1, )
        _, n = A.shape
        x = cp.Variable(n)

        obj = cp.Minimize(.5 * cp.sum_squares(A @ x - b))
        prob = cp.Problem(obj, [x >= 0])
        res = prob.solve()
        print(np.round(x.value, 4))

        unconstrained_prob = cp.Problem(obj)
        unconstrained_res = unconstrained_prob.solve()
        print(np.round(x.value, 4))
        print('constrained vs unconstrained res:')
        print(res, unconstrained_res)


def main():
    m, n = 60, 40
    # m, n = 3, 2
    b_c = 20 * np.ones((m, 1))
    b_r = 1
    instance = NNLS(m, n, b_c, b_r, seed=1)
    print(instance.mu, instance.L, instance.kappa)
    # print(instance.A)
    # instance.test_center_cvxpy()

if __name__ == '__main__':
    main()
