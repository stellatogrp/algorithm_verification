import cvxpy as cp
import numpy as np
import pandas as pd
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


def robust_param_set(n, x_init, xmin, xmax, **kwargs):
    num_samples = kwargs['num_samples']
    samples = np.random.uniform(low=xmin, high=xmax, size=(num_samples, n))
    sample_xmax = np.max(samples, axis=0)
    sample_xmin = np.min(samples, axis=0)
    return BoxSet(x_init, sample_xmin.reshape((-1, 1)), sample_xmax.reshape((-1, 1)))


def test_control_gen(n, N=1, t=.05, T=5):
    example = generate_problem_data(n)
    # q = example.qp_problem['q']
    A = example.qp_problem['A']
    # lx = example.qp_problem['lx']
    # ux = example.qp_problem['ux']
    l = example.qp_problem['l']
    u = example.qp_problem['u']
    print(A.shape)
    print(example.xmin)
    print(example.xmax)
    # print(lx, ux)
    print(l, u)
    # print(A)


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


def sample_xinit(xmin, xmax):
    np.random.seed(2)
    return np.random.uniform(xmin, xmax)


def run_admm(P, A, l, u, N=300):
    rho = 1
    rho_inv = 1 / rho
    sigma = 1
    m, n = A.shape
    ATA = A.T @ A
    In = np.eye(n)
    # Im = np.eye(m)

    def Pi(x):
        return np.minimum(u, np.maximum(x, l))

    xk = np.zeros(n)
    yk = np.zeros(m)
    zk = np.zeros(m)
    sk = zk + rho_inv * yk
    x_iterates = [xk]
    y_iterates = [yk]
    z_iterates = [zk]
    s_iterates = [sk]

    lhs = P + sigma * In + rho * ATA
    for _ in range(N):
        rhs = sigma * xk + A.T @ (rho * zk - yk)
        xkplus1 = np.linalg.solve(lhs, rhs)
        ykplus1 = yk + rho * (A @ xkplus1 - zk)
        zkplus1 = Pi(A @ xkplus1 + rho_inv * ykplus1)
        skplus1 = zkplus1 + rho_inv * ykplus1

        x_iterates.append(xkplus1)
        y_iterates.append(ykplus1)
        z_iterates.append(zkplus1)
        s_iterates.append(skplus1)

        xk = xkplus1
        yk = ykplus1
        zk = zkplus1
        sk = skplus1
    fixed_point_resids = []
    for i in range(1, N+1):
        x_resid = np.linalg.norm(x_iterates[i] - x_iterates[i-1]) ** 2
        s_resid = np.linalg.norm(s_iterates[i] - s_iterates[i-1]) ** 2
        fixed_point_resids.append(x_resid + s_resid)
    # print('tested for:', np.round(l, 3))
    # print(np.round(fixed_point_resids, 3))
    # print('direct admm x:', np.round(x_iterates, 4))
    # print('direct admm y:', np.round(y_iterates, 3))
    # print('direct admm z:', np.round(z_iterates, 3))
    return x_iterates, y_iterates, z_iterates, fixed_point_resids


def control_cert_prob_ws(n, N=1):
    example = generate_problem_data(n)
    A = example.qp_problem['A']
    full_m, full_n = A.shape
    P = example.qp_problem['P']
    l_ex = example.qp_problem['l']
    u_ex = example.qp_problem['u']
    # xmin = example.xmin
    # xmax = example.xmax
    # xinit_test = sample_xinit(xmin, xmax)
    # # print(xinit_test)
    # l = l_ex.copy()
    # u = u_ex.copy()
    # # # print(l, u, xmin, xmax)
    # l[:n] = -xinit_test
    # u[:n] = -xinit_test
    l = l_ex
    u = u_ex

    # solve problem optimally with cvxpy
    x_var = cp.Variable(full_n)
    obj = .5 * cp.quad_form(x_var, P)
    constraints = [l <= A @ x_var, A @ x_var <= u]
    problem = cp.Problem(cp.Minimize(obj), constraints)
    res = problem.solve()
    print('cvxpy result:', res)
    x_val = x_var.value
    z_val = A @ x_val

    xs, ys, zs, fixed_pt_resids = run_admm(P, A, l, u, N=1000)
    print(fixed_pt_resids[-1])
    x_last = xs[-1]
    print('admm result:', .5 * x_last.T @ P @ x_last)
    print('last y:', np.round(ys[-1], 4))
    # exit(0)
    # y_val = ys[-1]

    def iterate_set_func(x, y, z):
        zeros_fm = np.zeros((full_m, 1))
        xset = ConstSet(x, x_val.reshape(-1, 1))
        # yset = ConstSet(y, y_val.reshape(-1, 1))
        yset = ConstSet(y, zeros_fm)
        zset = ConstSet(z, z_val.reshape(-1, 1))
        # zset = ConstSet(z, zeros_fm)
        return xset, yset, zset

    return control_cert_prob(n, example, iter_set_func=iterate_set_func, N=N)


def control_cert_prob_robust_param(n, example, num_samples, N=1):
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

    return control_cert_prob(n, example, iter_set_func=iterate_set_func, xinit_set_func=robust_param_set,
                             N=N, num_samples=num_samples)


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


def run_and_save_experiments(max_N=2):
    # m = 4
    n = 2
    example = generate_problem_data(n)
    # N = 1
    # control_cert_prob(n, m, N=N)
    # test_control_gen(n)
    # control_cert_prob(n, N=N)
    save_dir = '/home/vranjan/algorithm-certification/experiments/control/data/'
    res_fname = save_dir + 'testN9_x0box.csv'
    x_fname = save_dir + 'test_xinitN9_x0box.csv'

    iterate_rows = []
    xinit_vals = []
    xinit_rows = []
    for N in range(1, max_N+1):
        (global_res, comp_time), xinit_res = control_cert_prob_non_ws(n, example, N=N)
        iter_row = pd.Series(
            {
                'num_iter': N,
                'global_res': global_res,
                'global_comp_time': comp_time,
            }
        )
        iterate_rows.append(iter_row)
        df = pd.DataFrame(iterate_rows)
        print(df)
        xinit_vals.append((N, tuple(np.round(xinit_res, 3))))
        df.to_csv(res_fname, index=False)
        xinit_row = pd.Series(
            {
                'num_iter': N,
                'x_init': xinit_res,
            }
        )
        xinit_rows.append(xinit_row)
        df2 = pd.DataFrame(xinit_rows)
        print(df2)
        df2.to_csv(x_fname, index=False)
        # print(xinit_res)
    # print(xinit_vals)


def run_and_save_ws_experiments(max_N=2):
    n = 2
    save_dir = '/home/vranjan/algorithm-certification/experiments/control/data/'
    res_fname = save_dir + 'testWSN9.csv'
    x_fname = save_dir + 'testWSxinitN9.csv'

    ws_iter_rows = []
    ws_xinit_rows = []
    for N in range(1, max_N+1):
        (global_res, comp_time), xinit_res = control_cert_prob_ws(n, N=N)
        iter_row = pd.Series(
            {
                'num_iter': N,
                'global_res': global_res,
                'global_comp_time': comp_time,
            }
        )
        ws_iter_rows.append(iter_row)
        df = pd.DataFrame(ws_iter_rows)
        print(df)
        df.to_csv(res_fname, index=False)
        # ws_xinit_rows.append((N, tuple(np.round(xinit_res, 3))))
        xinit_row = pd.Series(
            {
                'num_iter': N,
                'x_init': xinit_res,
            }
        )
        ws_xinit_rows.append(xinit_row)
        df2 = pd.DataFrame(ws_xinit_rows)
        print(df2)
        df2.to_csv(x_fname, index=False)


def run_and_save_robust_experiments(max_N=1):
    n = 2
    example = generate_problem_data(n)
    # save_dir = '/home/vranjan/algorithm-certification/experiments/control/data/'
    save_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/control/data'
    res_fname = save_dir + 'testrobust.csv'
    num_sample_vals = [1, 5]

    rows = []
    for N in range(1, max_N+1):
        (global_res, comp_time), _ = control_cert_prob_non_ws(n, example, N=N)
        iter_row = pd.Series(
            {
                'num_iter': N,
                'num_samples': -1,
                'global_res': global_res,
                'global_comp_time': comp_time,
            }
        )
        rows.append(iter_row)
        for num_samples in num_sample_vals:
            (global_res, comp_time), _ = control_cert_prob_robust_param(n, example, num_samples=num_samples, N=N)
            iter_row = pd.Series(
                {
                    'num_iter': N,
                    'num_samples': num_samples,
                    'global_res': global_res,
                    'global_comp_time': comp_time,
                }
            )
            rows.append(iter_row)
    df = pd.DataFrame(rows)
    print(df)
    df.to_csv(res_fname, index=False)


def main():
    n = 2
    example = generate_problem_data(n)
    print(example.x0, example.xmin, example.xmax)
    print(example.qp_problem['l'])
    # control_cert_prob_robust_param(n, example, 10, xinit_set_func=get_xinit_set)
    max_N = 3
    # control_cert_prob_non_ws(n, example, N=max_N)
    # control_cert_prob_robust_param(n, example, num_samples=5, N=max_N)
    # max_N = 1
    # run_and_save_experiments(max_N=max_N)
    # run_and_save_ws_experiments(max_N=max_N)
    run_and_save_robust_experiments(max_N=max_N)


if __name__ == '__main__':
    main()
