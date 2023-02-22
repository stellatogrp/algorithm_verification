import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as spa

from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.init_set.box_set import BoxSet
# from algocert.init_set.box_stack_set import BoxStackSet
# from algocert.init_set.centered_l2_ball_set import CenteredL2BallSet
from algocert.init_set.const_set import ConstSet
# from algocert.init_set.control_example_set import ControlExampleSet
# from algocert.init_set.init_set import InitSet
from algocert.objectives.convergence_residual import ConvergenceResidual
# from algocert.utils.plotter import plot_results
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


class NetworkUtilMax(object):
    def __init__(self, m_orig, n, K=1, seed=42, minimize=False):
        """
        min w^T f
            s.t. Rf <= c(theta)
                 0 <= f <= t
        R \in R^{m, n}
        """
        self.minimize = minimize
        np.random.seed(seed)

        # generate the problem data
        self.n, self.m_orig = n, m_orig
        self.m = m_orig + 2 * n
        z_size = self.m + self.n
        self.z_size = z_size

        self.R = np.random.binomial(n=1, p=0.2, size=(self.m_orig, self.n))
        self.c_sample = np.random.uniform(0, 1, size=self.m)
        # self.s_sample = np.random.uniform(.1, 1, size=self.n)
        self.w_sample = np.random.uniform(0, 1, size=self.n)
        self.t_sample = 1 * np.ones(self.n)

        # iterates and parameter setup
        u = Iterate(z_size, name='u')
        u_tilde = Iterate(z_size, name='u_tilde')
        v = Iterate(z_size, name='v')
        z = Iterate(z_size, name='z')
        q = Parameter(z_size, name='q')

        # canonicalize and get the matrix M and set for the Parameter q
        M, lower, upper = self.canonicalize()

        # setup constants
        zeros = np.zeros((z_size, 1))
        I = np.eye(z_size)
        spI = spa.csc_matrix(I)

        # step 1 (M + I)u = z - q
        # This step encodes u = (M + I)^{-1} (z-q)
        s1D = spI
        s1A_temp = spa.bmat([[spI, -I]])
        MIinv = spa.csc_matrix(np.linalg.inv(M + np.eye(z_size)))
        s1A = MIinv @ s1A_temp
        step1 = HighLevelLinearStep(u, [z, q], D=s1D, A=s1A, b=zeros, Dinv=s1D)

        # # alt step 1 (THIS ALSO WORKS)
        # This encodes (M + I)u = (z-q)
        # s1D = spa.csc_matrix(M + np.eye(z_size))
        # s1Dinv = spa.csc_matrix(np.linalg.inv(M + np.eye(z_size)))
        # s1A = spa.bmat([[spI, -I]])
        # step1 = HighLevelLinearStep(u, [z, q], D=s1D, A=s1A, b=zeros, Dinv=s1Dinv)

        # step 2 (v = 2u - z)
        s2D = spI
        s2A = spa.bmat([[2 * spI, -I]])
        step2 = HighLevelLinearStep(v, [u, z], D=s2D, A=s2A, b=zeros, Dinv=spI)

        # step 3 (u_tilde = Pi(v))
        step3 = MaxWithVecStep(u_tilde, v, zeros)

        # step 4 (z = z + u_tilde - u)
        s4D = spI
        s4A = spa.bmat([[spI, -I, I]])
        step4 = HighLevelLinearStep(z, [z, u, u_tilde], D=s4D, A=s4A, b=zeros, Dinv=spI)

        self.steps = [step1, step2, step3, step4]
        self.zset = ConstSet(z, np.zeros((z_size, 1)))
        self.qset = BoxSet(q, lower, upper)
        self.obj = [ConvergenceResidual(z)]

    def canonicalize(self):
        I = np.eye(self.n)

        # form A
        A = np.zeros((self.m, self.n))
        A[:self.m_orig, :] = self.R
        A[self.m_orig: self.m_orig + self.n, :] = I
        A[self.m_orig + self.n:, :] = -I

        # form M
        M = np.zeros((self.z_size, self.z_size))
        M[:self.n, self.n:] = A.T
        M[self.n:, :self.n] = -A

        # form the box set for q
        l = np.zeros((self.z_size, 1))
        u = np.zeros((self.z_size, 1))
        # u = 5 * np.ones((self.z_size, 1))

        # update l, u
        l[:self.n, 0] = self.w_sample
        u[:self.n, 0] = self.w_sample

        # this is for the parameters c(theta)
        u[self.n:self.n + self.m_orig, 0] = 0.5

        l[self.n + self.m_orig: 2 * self.n + self.m_orig, 0] = self.t_sample
        u[self.n + self.m_orig: 2 * self.n + self.m_orig, 0] = self.t_sample

        return M, l, u

    def solve(self, K, solve_type, solver_args={}, verbose=True):
        CP = CertificationProblem(K, [self.zset], [self.qset], self.obj, self.steps)
        if solve_type == 'SDP':
            add_RLT = solver_args.get('add_RLT', True)
            solver = solver_args.get('solver', cp.MOSEK)
            res = CP.solve(solver_type='SDP',
                           solver=solver,
                           add_RLT=add_RLT,
                           verbose=verbose,
                           minimize=self.minimize,
                           )
        elif solve_type == 'GLOBAL':
            time_limit = solver_args.get('time_limit', 3600)
            add_bounds = solver_args.get('add_bounds', True)
            res = CP.solve(solver_type='GLOBAL', add_bounds=add_bounds,
                           TimeLimit=time_limit, verbose=verbose, minimize=self.minimize)
        else:
            print('check solver type')
        return res


def experiment():
    # save_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/NUM/data/'
    # fname = save_dir + 'only5.csv'
    m, n = 2, 4
    np.random.seed(0)

    K_vals = [1, 2, 3, 4, 5]
    K_vals = [5]

    res_rows = []
    for K in K_vals:
        print('K:', K)
        NUM = NetworkUtilMax(m, n, K=K)
        res_g = NUM.solve(K, 'GLOBAL')

        NUM = NetworkUtilMax(m, n, K=K)
        res_sdp = NUM.solve(K, 'SDP')
        res_row = pd.Series(
            {
                'K': K,
                'g_obj': res_g[0],
                'g_solve_time': res_g[1],
                'sdp_obj': res_sdp[0],
                'sdp_solve_time': res_sdp[1],
                'min_max': 'max',
            }
        )
        res_rows.append(res_row)

        NUM = NetworkUtilMax(m, n, K=K, minimize=True)
        res_g = NUM.solve(K, 'GLOBAL')

        NUM = NetworkUtilMax(m, n, K=K, minimize=True)
        # res_sdp = NUM.solve('SDP')
        res_sdp = [-1, -1]
        res_row = pd.Series(
            {
                'K': K,
                'g_obj': res_g[0],
                'g_solve_time': res_g[1],
                'sdp_obj': res_sdp[0],
                'sdp_solve_time': res_sdp[1],
                'min_max': 'min',
            }
        )
        res_rows.append(res_row)

        df = pd.DataFrame(res_rows)
        # df.to_csv(fname, index=False)
    print(df)
    # for K in K_vals:
    #     print('K:', K)
    #     newtork_util_max = NetworkUtilMax(m, n, K=K)
    #     res_sdp = newtork_util_max.solve('SDP')

    #     res_srlt_row = pd.Series(
    #         {
    #             'num_iter': K,
    #             'obj': res_sdp[0],
    #             'solve_time': res_sdp[1],
    #         }
    #     )
    #     res_srlt_rows.append(res_srlt_row)

    #     res_global = newtork_util_max.solve('global')
    #     res_g_row = pd.Series(
    #         {
    #             'num_iter': K,
    #             'obj': res_global[0],
    #             'solve_time': res_global[1],
    #         }
    #     )
    #     res_g_rows.append(res_g_row)

    # df_srlt = pd.DataFrame(res_srlt_rows)
    # df_g = pd.DataFrame(res_g_rows)

    # print(df_srlt)
    # print(df_g)

    # srlt_fname = save_dir + 'sdp_rlt.csv'
    # g_fname = save_dir + 'global.csv'

    # df_srlt.to_csv(srlt_fname, index=False)
    # df_g.to_csv(g_fname, index=False)


def main():
    experiment()
    # NUM = NetworkUtilMax(2, 4, K=5)
    # NUM.solve('GLOBAL')
    # NUM.solve('SDP')


if __name__ == '__main__':
    main()
