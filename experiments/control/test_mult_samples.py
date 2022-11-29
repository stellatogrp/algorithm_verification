# import cvxpy as cp
import gurobipy as gp
import numpy as np
# import pandas as pd
import scipy.sparse as spa
# from control_example import ControlExample
from quadcopter import QuadCopter

from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
from algocert.certification_problem import CertificationProblem
from algocert.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algocert.init_set.box_set import BoxSet
from algocert.init_set.box_stack_set import BoxStackSet
from algocert.init_set.const_set import ConstSet
# from algocert.init_set.offcenter_l2_ball_set import OffCenterL2BallSet
# from algocert.init_set.control_example_set import ControlExampleSet
from algocert.objectives.convergence_residual import ConvergenceResidual
# from algocert.objectives.lin_comb_squared_norm import LinCombSquaredNorm
from algocert.variables.iterate import Iterate
from algocert.variables.parameter import Parameter


def generate_problem_data(T=5):
    return QuadCopter(T=T)


def get_samples(example, N_samples=10):
    nx = 6
    # nu = 3
    cvx_prob, (x, u), x0 = example._generate_cvxpy_problem()
    _ = cvx_prob.solve()
    x_prev = x.value
    u_prev = u.value
    A = example.A
    B = example.B
    # print(x_prev.shape, u_prev.shape)
    # print(np.round(x0.value, 4), np.round(x_prev[:, 0], 4))
    samples = []
    print('generating')
    for _ in range(N_samples):
        xk = x_prev[:, 0]
        # print(xk)
        uk = u_prev[:, 0]
        noise = np.random.normal(0, .001, nx)
        xnew = A @ xk + B @ uk
        # print(A.shape, xk.shape, B.shape, uk.shape)
        # print(np.round(xnew, 4), np.round(xnew+noise, 4)
        xnew = xnew + noise
        example.update_x0(xnew)
        print(np.round(xnew, 4))
        cvx_prob, (x, u), x0 = example._generate_cvxpy_problem()
        _ = cvx_prob.solve()
        # print(example.qp_problem['l'][:nx])
        x_prev = x.value
        u_prev = u.value
        prev_sol = np.append(x_prev.flatten('F'), u_prev.flatten('F'))
        samples.append(prev_sol)
    return samples


def experiment(example, max_N=1, N_samples=10):
    timelimit = 3600
    # eps = .05
    samples = get_samples(example, N_samples=N_samples)
    # print(samples)
    # print(example.qp_problem['P'].shape, example.qp_problem['A'].shape, example.qp_problem['l'].shape)
    model = make_gurobi_model(timelimit=timelimit)
    overall_obj = 0
    for i in range(N_samples):
        obj = form_CP(example, samples[i], model, N=max_N)
        overall_obj += obj
    model.setObjective(-overall_obj / N_samples, gp.GRB.MINIMIZE)
    model.optimize()
    print(model.Runtime)


def experiment_cvar(example, max_N=1, N_samples=10, xinit_eps=.01):
    timelimit = 10800
    eps = .05
    samples = get_samples(example, N_samples=N_samples)
    model = make_gurobi_model(timelimit=timelimit)
    summation = 0
    alpha = model.addVar(ub=0, lb=-5)
    for i in range(N_samples):
        obj = form_CP(example, samples[i], model, N=max_N, xinit_eps=xinit_eps)
        zi = model.addVar(ub=gp.GRB.INFINITY, lb=0)
        val = model.addVar(ub=gp.GRB.INFINITY, lb=-gp.GRB.INFINITY)
        model.addConstr(val == (-obj - alpha))
        model.addConstr(zi >= val)
        model.addConstr(zi * val == 0)
        summation += zi / eps
    overall_obj = summation / N_samples + alpha
    model.setObjective(overall_obj, gp.GRB.MINIMIZE)
    model.optimize()
    print(model.Runtime)


def make_gurobi_model(timelimit=3600):
    m = gp.Model()
    m.setParam('NonConvex', 2)
    m.setParam('MIPGap', .1)
    m.setParam('TimeLimit', timelimit)
    return m


def form_CP(quadcopter, xprev, gp_model, N=2, xinit_eps=.01):
    nx = 6
    # nu = 3

    # xinit_eps = .01
    # x_eps = .05
    rho = 1
    rho_inv = 1 / rho
    sigma = 1
    P = quadcopter.qp_problem['P']
    A = quadcopter.qp_problem['A']
    ATA = A.T @ A
    full_m, full_n = A.shape
    l = quadcopter.qp_problem['l']
    u = quadcopter.qp_problem['u']
    # xmin = quadcopter.xmin
    # xmax = quadcopter.xmax

    Ax_prev = A @ xprev
    Hx_prev = P @ xprev
    linstep_b = -Hx_prev
    print(Ax_prev.shape, Hx_prev.shape)
    print(np.round(Ax_prev, 4))
    # print(l.shape, Ax_prev.shape)

    l_shift = l - Ax_prev
    u_shift = u - Ax_prev
    # print(np.round(l_shift, 4))

    neg_xinit = l_shift[:nx]
    l_noinit = l_shift[nx:]
    u_noinit = u_shift[nx:]
    l_mat = l_noinit.reshape((-1, 1))
    u_mat = u_noinit.reshape((-1, 1))
    x_init = Parameter(nx, name='x_init')
    # create x_init set
    x_initset = BoxSet(x_init, (neg_xinit - xinit_eps).reshape((-1, 1)), (neg_xinit + xinit_eps).reshape((-1, 1)))

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

    s1_Dtemp = P + sigma * I_fn + rho * ATA
    s1_Atemp = spa.bmat([[sigma * I_fn, rho * A.T, -rho * A.T]])
    s1_D = I_fn
    s1_A = spa.csc_matrix(spa.linalg.inv(s1_Dtemp) @ s1_Atemp)
    step1 = HighLevelLinearStep(x, [x, z, y], D=s1_D, A=s1_A, b=linstep_b, Dinv=s1_D)

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

    zeros_fn = np.zeros((full_n, 1))
    # ones_fn = np.ones((full_n, 1))
    zeros_fm = np.zeros((full_m, 1))
    # xset = BoxSet(x, -x_eps * ones_fn, x_eps * ones_fn)
    xset = ConstSet(x, zeros_fn)
    # xset = OffCenterL2BallSet(x, zeros_fn, .01)
    yset = ConstSet(y, zeros_fm)
    zset = ConstSet(z, zeros_fm)
    obj = [ConvergenceResidual(x), ConvergenceResidual(s)]
    CP = CertificationProblem(N, [xset, yset, zset], paramsets, obj, steps)

    # resg = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600)
    # print('global', resg)
    solver = CP.canonicalize(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600, model=gp_model)
    handler = solver.handler
    obj = handler.objective
    # gp_model.setObjective(obj, gp.GRB.MAXIMIZE)
    # gp_model.setObjective(-obj)
    # handler.canonicalize()
    # gp_model.optimize()
    return obj


def main():
    np.random.seed(0)
    N_samples = 1
    max_N = 1
    xinit_eps = .01
    print('N_samples=', N_samples)
    print('max_N=', max_N)
    print('xinit_eps=', xinit_eps)
    example = generate_problem_data()
    # experiment(example, N_samples=N_samples)
    experiment_cvar(example, max_N=max_N, N_samples=N_samples, xinit_eps=xinit_eps)


if __name__ == '__main__':
    main()
