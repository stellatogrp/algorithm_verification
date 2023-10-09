import numpy as np
import scipy.sparse as spa

from algocert.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algocert.certification_problem import CertificationProblem
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

    def _generate_A_mat(self):
        np.random.seed(self.seed)
        self.A = np.random.randn(self.m, self.n)

    def get_t_vals(self):
        A = self.A
        ATA = A.T @ A
        eigs = np.linalg.eigvals(ATA)
        mu = np.min(eigs)
        L = np.max(eigs)
        return (1 / L), 2 / (mu + L), 2 / L

    def generate_CP(self, t, K):
        m, n = self.m, self.n
        A = spa.csc_matrix(self.A)
        ATA = A.T @ A
        In = spa.eye(n)

        # print((In - t * ATA).shape, (t * A.T).shape)
        C = spa.bmat([[In - t * ATA, t * A.T]])
        D = spa.eye(n, n)
        b_const = np.zeros((n, 1))

        y = Iterate(n, name='y')
        x = Iterate(n, name='x')
        b = Parameter(m, name='b')

        xset = ZeroSet(x)
        bset = L2BallSet(b, self.b_c, self.b_r)

        step1 = LinearStep(y, [x, b], D=D, A=C, b=b_const, Dinv=D)
        step2 = NonNegProjStep(x, y)

        steps = [step1, step2]
        var_sets = [xset]
        param_sets = [bset]
        obj = ConvergenceResidual(x)

        return CertificationProblem(K, var_sets, param_sets, obj, steps)


def main():
    m, n = 5, 3
    b_c = 10 * np.ones((m, 1))
    b_r = .1
    t = .05
    K = 3
    instance = NNLS(m, n, b_c, b_r, seed=1)
    NNLS(m, n, b_c, b_r, seed=1)
    CP = instance.generate_CP(t, K)
    CP2 = instance.generate_CP(t, K)
    t_vals = list(instance.get_t_vals())
    # (sdp_c, sdp_ctime) = CP.solve(solver_type='SDP_CUSTOM')
    # (sdp_g, sdp_gtime) = CP2.solve(solver_type='GLOBAL', add_bounds=True)
    # print(sdp_c, sdp_g)
    # print(sdp_ctime, sdp_gtime)
    out_c = []
    out_g = []
    instance = NNLS(m, n, b_c, b_r, seed=1)
    t_vals.append(.05)
    for t in t_vals:
        CP = instance.generate_CP(t, K)
        sdp_g, sdp_gtime = CP.solve(solver_type='GLOBAL', add_bounds=True)
        out_g.append(sdp_g)
        CP2 = instance.generate_CP(t, K)
        out = CP2.solve(solver_type='SDP_CUSTOM')
        sdp_c, sdp_ctime = out['sdp_objval'], out['sdp_solvetime']
        out_c.append(sdp_c)
        print(sdp_ctime)


if __name__ == '__main__':
    main()
