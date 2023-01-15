# import cvxpy as cp
import numpy as np
# import pandas as pd
import scipy.sparse as spa
from control_example import ControlExample

from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.init_set.box_set import BoxSet
from algocert.init_set.box_stack_set import BoxStackSet
from algocert.init_set.const_set import ConstSet
# from algocert.init_set.control_example_set import ControlExampleSet
from algocert.objectives.convergence_residual import ConvergenceResidual
# from algocert.objectives.lin_comb_squared_norm import LinCombSquaredNorm
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


def generate_problem_data(n):
    return ControlExample(n)


def get_xinit_set(n, x_init, xmin, xmax, **kwargs):
    return BoxSet(x_init, xmin.reshape((-1, 1)), xmax.reshape((-1, 1)))
    # test = np.array([1.43069857, 1.93912779])
    # return BoxSet(x_init, test.reshape((-1, 1)), test.reshape((-1, 1)))


def control_cert_prob_non_ws(n, example, N=1):
    # example = generate_problem_data(n)
    A = example.qp_problem['A']
    full_m, full_n = A.shape

    def iterate_set_func(x, y, z):
        zeros_fn = np.zeros((full_n, 1))
        ones_fn = np.ones((full_n, 1))
        zeros_fm = np.zeros((full_m, 1))
        # z_val = A @ zeros_fn
        # xset = ConstSet(x, zeros_fn)
        xset = BoxSet(x, zeros_fn, ones_fn)
        yset = ConstSet(y, zeros_fm)
        zset = ConstSet(z, zeros_fm)
        # zset = ConstSet(z, z_val.reshape(-1, 1))
        return xset, yset, zset

    return control_cert_prob(n, example, iter_set_func=iterate_set_func, xinit_set_func=get_xinit_set, N=N)


def control_cert_prob(n, example, iter_set_func=None, xinit_set_func=None, N=1, num_samples=None):
    # example = generate_problem_data(n)
    rho = 1
    rho_inv = 1 / rho
    sigma = 1
    P = example.qp_problem['P']
    A = example.qp_problem['A']
    ATA = A.T @ A
    full_m, full_n = A.shape
    l = example.qp_problem['l']
    u = example.qp_problem['u']
    xmin = example.xmin
    xmax = example.xmax
    # print(xmin, xmax)
    # print(np.linalg.eigvals(ATA.todense()))
    # exit(0)

    # print(l, u)
    l_noinit = l[n:]
    u_noinit = u[n:]
    l_mat = l_noinit.reshape((-1, 1))
    u_mat = u_noinit.reshape((-1, 1))
    x_init = Parameter(n, name='x_init')
    # x_initset = BoxSet(x_init, xmin.reshape((-1, 1)), xmax.reshape((-1, 1)))
    # x_initset = get_xinit_set(n, x_init, xmin, xmax)
    # x_initset = get_xinit_set(n, x_init, -xmax, -xmin)
    x_initset = xinit_set_func(n, x_init, xmin, xmax, num_samples=num_samples)

    l_param = Parameter(full_m, name='l_param')
    l_paramset = BoxStackSet(l_param, [x_initset, [l_mat, l_mat]])

    u_param = Parameter(full_m, name='u_param')
    u_paramset = BoxStackSet(u_param, [x_initset, [u_mat, u_mat]])

    paramsets = [x_initset, l_paramset, u_paramset]

    x = Iterate(full_n, name='x')
    y = Iterate(full_m, name='y')
    w = Iterate(full_m, name='w')
    z_tilde = Iterate(full_m, name='z_tilde')
    z = Iterate(full_m, name='z')

    I_fn = spa.eye(full_n)
    I_fm = spa.eye(full_m)
    zeros_fn = np.zeros((full_n, 1))
    zeros_fm = np.zeros((full_m, 1))

    # step 1
    s1_Dtemp = P + sigma * I_fn + rho * ATA
    s1_Atemp = spa.bmat([[sigma * I_fn, rho * A.T, -rho * A.T]])
    s1_D = I_fn
    s1_A = spa.csc_matrix(spa.linalg.inv(s1_Dtemp) @ s1_Atemp)
    step1 = HighLevelLinearStep(x, [x, z, y], D=s1_D, A=s1_A, b=zeros_fn, Dinv=s1_D)

    # step 2
    s2_D = I_fm
    s2_A = spa.bmat([[I_fm, rho * A, -rho * I_fm]])
    step2 = HighLevelLinearStep(y, [y, x, z], D=s2_D, A=s2_A, b=zeros_fm, Dinv=s2_D)

    # step 3
    s3_D = I_fm
    s3_A = spa.bmat([[A, rho_inv * I_fm]])
    step3 = HighLevelLinearStep(w, [x, y], D=s3_D, A=s3_A, b=zeros_fm, Dinv=s3_D)

    # step 4
    step4 = MaxWithVecStep(z_tilde, w, l=l_param)

    # step 5
    step5 = MinWithVecStep(z, z_tilde, u=u_param)


#     # step 6 for fixed point residual
    s = Iterate(full_m, name='s')
    s6_D = I_fm
    s6_A = spa.bmat([[I_fm, rho_inv * I_fm]])
    step6 = HighLevelLinearStep(s, [z, y], D=s6_D, A=s6_A, b=zeros_fm, Dinv=s6_D)
#
    steps = [step1, step2, step3, step4, step5, step6]
#
#     # xset = CenteredL2BallSet(x, r=r)
#     x_l = -1 * np.ones((n, 1))
#     x_u = np.ones((n, 1))
#     xset = BoxSet(x, x_l, x_u)
    # xset = ConstSet(x, zeros_fn)
    # yset = ConstSet(y, zeros_fm)
    # zset = ConstSet(z, zeros_fm)
    xset, yset, zset = iter_set_func(x, y, z)

    # obj = [ConvergenceResidual(x), ConvergenceResidual(y), ConvergenceResidual(z)]
    obj = [ConvergenceResidual(x), ConvergenceResidual(s)]
#     # obj = OuterProdTrace(x)
    # obj_A = [A, -I_fm]
    # obj_x = [x, z]
    # obj1 = LinCombSquaredNorm(obj_A, obj_x)

    # obj_B = [P, A.T]
    # obj_y = [x, y]
    # obj2 = LinCombSquaredNorm(obj_B, obj_y)

    # obj = [obj1, obj2]

    CP = CertificationProblem(N, [xset, yset, zset], paramsets, obj, steps)
#
#     CP.print_cp()
#
#     # res = CP.solve(solver_type='SDP', add_RLT=False, verbose=True)
#     # print('sdp', res)
#     # res = CP.solve(solver_type='SDP', add_RLT=True, verbose=True)
#     # print('sdp rlt', res)
    resg = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600)
    print('global', resg)
    xinit_res = None
    # param_map = CP.get_param_map()
    # xinit_res = param_map[x_init].X
    # print('xinit val:', xinit_res)
    # print('testing l:', np.round(param_map[l_param].X, 3))
    # print('l val:', CP.get_param_map()[l_param].X)
    # print('u val:', CP.get_param_map()[u_param].X)

    # iter_map = CP.get_iterate_map()
    # print('test file x:', np.round(iter_map[x].X, 4))
    # print('test file y:', np.round(iter_map[y].X, 3))
    # print('test file z:', np.round(iter_map[z].X, 3))
    return resg, xinit_res


def main():
    n = 2
    example = generate_problem_data(n)

    max_N = 1
    control_cert_prob_non_ws(n, example, N=max_N)


if __name__ == '__main__':
    main()
