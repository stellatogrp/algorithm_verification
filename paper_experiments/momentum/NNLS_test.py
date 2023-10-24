from datetime import datetime

import numpy as np
import pandas as pd
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

        step1 = LinearStep(y, [x, b], D=D, A=C, b=b_const, Dinv=D)
        step2 = NonNegProjStep(x, y)

        steps = [step1, step2]
        var_sets = [xset]
        param_sets = [bset]
        obj = ConvergenceResidual(x)

        return CertificationProblem(K, var_sets, param_sets, obj, steps)


def generate_all_t_vals(t_vals, num_between=1):
    t_min, t_opt, t_max = t_vals
    t_min_to_opt = np.logspace(np.log10(t_min), np.log10(t_opt), num=num_between+1)
    t_opt_to_max = np.logspace(np.log10(t_opt), np.log10(t_max), num=num_between+1)
    # print(t_min_to_opt)
    # print(t_opt_to_max)
    t_out = np.hstack([t_min_to_opt, t_opt_to_max[1:]])
    print(t_out)
    return t_out


def main():
    d = datetime.now()
    # print(d)
    curr_time = d.strftime('%m%d%y_%H%M%S')
    # outf_prefix = '/home/vranjan/algorithm-certification/'
    outf_prefix = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/'
    outf = outf_prefix + f'paper_experiments/NNLS/data/{curr_time}.csv'
    print(outf)

    # m, n = 30, 15
    m, n = 3, 2
    b_c = 10 * np.ones((m, 1))
    b_r = 1
    # K = 5
    K_vals = [3]
    # K_vals = [9, 10]
    # K_vals = [7, 8]
    # K_vals = [1, 2, 3, 4, 5, 6]

    instance = NNLS(m, n, b_c, b_r, seed=1)

    # t_vals = list(instance.get_t_vals())
    # t_vals = generate_all_t_vals(t_vals)
    # print(t_vals)
    t_vals = [[.05, .06, .07]]

    out_res = []
    for K in K_vals:
        for t in t_vals:
        # t = t_vals[1]
            CP = instance.generate_CP(t, K)
            out = CP.solve(solver_type='SDP_CUSTOM')
            out['orig_m'] = m
            out['orig_n'] = n
            out['b_r'] = b_r
            out['K'] = K
            out['t'] = t
            sdp_c, sdp_canontime, sdp_solvetime = out['sdp_objval'], out['sdp_canontime'], out['sdp_solvetime']
            print(sdp_c, sdp_canontime, sdp_solvetime)

            # out_df = []
            out_res.append(pd.Series(out))
            out_df = pd.DataFrame(out_res)


            CP2 = instance.generate_CP(t, K)
            outg, outgtime = CP2.solve(solver_type='GLOBAL', add_bounds=True)

            print(out_df)
            print(outg, outgtime)
            # out_df.to_csv(outf, index=False)


if __name__ == '__main__':
    main()
