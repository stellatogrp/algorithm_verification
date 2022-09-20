import unittest

import cvxpy as cp
import numpy as np
import numpy.testing as npt
import scipy.sparse as spa

from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
# from algocert.init_set.affine_vec_set import AffineVecSet
from algocert.init_set.box_set import BoxSet
from algocert.init_set.const_set import ConstSet
from algocert.objectives.convergence_residual import ConvergenceResidual
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


class TestBasicGD(unittest.TestCase):

    def setUp(self):
        m = 3
        n = 2
        N = 1
        self.m = m
        self.n = n
        self.N = N
        self.t = .05

        np.random.seed(0)
        Hhalf = np.random.randn(n, n)
        self.H = Hhalf @ Hhalf.T

        self.S = np.random.randn(m, n)
        self.b = np.random.randn(m, 1)

        self.theta = Parameter(n, name='theta')
        self.q = Parameter(m, name='q')

        self.theta_l = -1 * np.ones((n, 1))
        self.theta_u = np.ones((n, 1))

    def test_param_max_proj(self):
        n, N, t = self.n, self.N, self.t
        H = self.H
        b, b_l, b_u = self.theta, self.theta_l, self.theta_u
        bset = BoxSet(b, b_l, b_u)

        In = spa.eye(n)
        C = In - t * H
        C = spa.csc_matrix(C)
        zeros_n = np.zeros((n, 1))

        x = Iterate(n, name='x')
        y = Iterate(n, name='y')

        xset = ConstSet(x, zeros_n)

        # step1 = HighLevelLinearStep(x, [x, z, y, b], D=s1_D, A=s1_A, b=zeros_n, Dinv=s1_D)
        step1 = HighLevelLinearStep(y, [x], D=In, A=C, b=zeros_n, Dinv=In)
        step2 = MaxWithVecStep(x, y, b)
        steps = [step1, step2]

        obj = ConvergenceResidual(x)
        # obj = OuterProdTrace(x)

        CP = CertificationProblem(N, [xset], [bset], obj, steps)
        res_global = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=100)
        print('global', res_global)
        res = CP.solve(solver_type='SDP', add_RLT=False, verbose=False, solver=cp.SCS)
        res_RLT = CP.solve(solver_type='SDP', add_RLT=True, verbose=False, solver=cp.SCS)
        print('normal', res)
        print('RLT', res_RLT)
        npt.assert_array_less([res_RLT[0]], [res[0]])  # this might be problematic with numerical precision

    def test_brute_force_max_proj(self):
        # m, n, N, t = self.m, self.n, self.N, self.t
        # H = self.H
        # b_l, b_u = self.theta_l, self.theta_u
        pass
