import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt
import scipy.sparse as spa

from algoverify.high_level_alg_steps.linear_step import LinearStep
from algoverify.init_set.box_set import BoxSet
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.variables.iterate import Iterate
from algoverify.variables.parameter import Parameter
from algoverify.verification_problem import VerificationProblem


class TestBasicGD(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.t = .05
        self.n = 2
        In = spa.eye(self.n)
        Phalf = np.random.randn(self.n, self.n)
        P = spa.csc_matrix(Phalf @ Phalf)
        self.C = spa.bmat([[In - self.t * P, -self.t * In]])
        self.x_l = -1 * np.ones((self.n, 1))
        self.x_u = np.ones((self.n, 1))
        self.q_l = np.ones((self.n, 1))
        self.q_u = 3 * np.ones((self.n, 1))

    def test_CP_vs_brute_force_SDP(self):
        N = 1
        n, C, x_l, x_u, q_l, q_u = self.n, self.C, self.x_l, self.x_u, self.q_l, self.q_u

        # first run the brute force SDP
        q = cp.Variable((n, 1))
        qqT = cp.Variable((n, n))
        x0 = cp.Variable((n, 1))
        x0x0T = cp.Variable((n, n))
        x0qT = cp.Variable((n, n))

        constraints = [
            cp.reshape(cp.diag(x0x0T), (n, 1)) <= cp.multiply((x_l + x_u), x0) - x_l * x_u,
            cp.reshape(cp.diag(qqT), (n, 1)) <= cp.multiply((q_l + q_u), q) - q_l * q_u,
        ]

        u1 = cp.vstack([x0, q])
        u1u1T = cp.bmat([
            [x0x0T, x0qT],
            [x0qT.T, qqT]
        ])
        u1x0T = cp.bmat([
            [x0x0T],
            [x0qT]
        ])

        x1 = C @ u1
        x1x1T = C @ u1u1T @ C.T
        x1x0T = C @ u1x0T

        constraints += [
            cp.bmat([
                [u1u1T, u1],
                [u1.T, np.array([[1]])]
            ]) >> 0,
            cp.bmat([
                [x1x1T, x1x0T, x1],
                [x1x0T.T, x0x0T, x0],
                [x1.T, x0.T, np.array([[1]])]
            ]) >> 0,
        ]
        obj = cp.trace(x1x1T - 2 * x1x0T + x0x0T)
        prob = cp.Problem(cp.Maximize(obj), constraints)
        brute_force_res = prob.solve(solver=cp.SCS)

        # then, use a CP instead
        b_const = np.zeros(n)
        In = spa.eye(n)

        x = Iterate(n, name='x')
        q = Parameter(n, name='q')

        step1 = LinearStep(x, [x, q], D=In, A=C, b=b_const, Dinv=In)

        x_l = -1 * np.ones((n, 1))
        x_u = np.ones((n, 1))
        xset = BoxSet(x, x_l, x_u)

        q_l = np.ones((n, 1))
        q_u = 3 * np.ones((n, 1))
        qset = BoxSet(q, q_l, q_u)

        obj = ConvergenceResidual(x)
        CP = VerificationProblem(N, [xset], [qset], obj, [step1])

        CP_res, _ = CP.solve(solver_type='SDP_CUSTOM', solver=cp.SCS)
        npt.assert_allclose(brute_force_res, CP_res, rtol=1e-4, atol=1e-4)
