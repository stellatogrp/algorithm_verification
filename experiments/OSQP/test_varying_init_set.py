import cvxpy as cp
import gurobipy as gp
import numpy as np
import pandas as pd
import scipy.sparse as spa

from algoverify.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algoverify.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
from algoverify.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algoverify.init_set.box_set import BoxSet

# from algoverify.init_set.box_stack_set import BoxStackSet
# from algoverify.init_set.centered_l2_ball_set import CenteredL2BallSet
from algoverify.init_set.const_set import ConstSet

# from algoverify.init_set.control_example_set import ControlExampleSet
# from algoverify.init_set.init_set import InitSet
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.variables.iterate import Iterate
from algoverify.variables.parameter import Parameter
from algoverify.verification_problem import VerificationProblem


def OSQP_CP_noboxstack(n, m, N=1, eps_b=.1, b_sample=None, r_x=1, solver_type='SDP', add_RLT=True,
                       verbose=False, gp_model=None):
    #  r = 1

    In = spa.eye(n)
    Im = spa.eye(m)

    np.random.seed(0)
    A = np.random.randn(m, n)
    A = spa.csc_matrix(A)
    ATA = A.T @ A
    Phalf = np.random.randn(n, n)
    P = Phalf.T @ Phalf
    # print(A)

    # b_const = spa.csc_matrix(np.zeros((n, 1)))
    zeros_n = np.zeros((n, 1))
    zeros_m = np.zeros((m, 1))
    l = 2 * np.ones((m, 1))
    u = 4 * np.ones((m, 1))
    sigma = 1
    rho = 1
    rho_inv = 1 / rho

    x = Iterate(n, name='x')
    y = Iterate(m, name='y')
    w = Iterate(m, name='w')
    z_tilde = Iterate(m, name='z_tilde')
    z = Iterate(m, name='z')
    b = Parameter(n, name='b')

    # step 1
    s1_Dtemp = P + sigma * In + rho * ATA
    s1_Atemp = spa.bmat([[sigma * In, rho*A.T, -rho * A.T, -In]])
    s1_D = In
    s1_A = spa.csc_matrix(np.linalg.inv(s1_Dtemp) @ s1_Atemp)
    step1 = HighLevelLinearStep(x, [x, z, y, b], D=s1_D, A=s1_A, b=zeros_n, Dinv=s1_D)

    # step 2
    s2_D = Im
    s2_A = spa.bmat([[Im, rho * A, rho * Im]])
    step2 = HighLevelLinearStep(y, [y, x, z], D=s2_D, A=s2_A, b=zeros_m, Dinv=s2_D)

    # step 3
    s3_D = Im
    s3_A = spa.bmat([[A, 1/rho * Im]])
    step3 = HighLevelLinearStep(w, [x, y], D=s3_D, A=s3_A, b=zeros_m, Dinv=s3_D)

    # step 4
    step4 = MaxWithVecStep(z_tilde, w, l=l)

    # step 5
    step5 = MinWithVecStep(z, z_tilde, u=u)

    # step 6 for fixed point residual
    s = Iterate(m, name='s')
    s6_D = Im
    s6_A = spa.bmat([[Im, rho_inv * Im]])
    step6 = HighLevelLinearStep(s, [z, y], D=s6_D, A=s6_A, b=zeros_m, Dinv=s6_D)

    # steps = [step1, step2, step3, step4, step5]
    steps = [step1, step2, step3, step4, step5, step6]

    # xset = CenteredL2BallSet(x, r=r_x)
    x_l = -r_x * np.ones((n, 1))
    x_u = r_x * np.ones((n, 1))
    xset = BoxSet(x, x_l, x_u)
    # xset = ConstSet(x, np.zeros((n, 1)))

    yset = ConstSet(y, np.zeros((m, 1)))

    zset = ConstSet(z, np.zeros((m, 1)))

    # b_l = - eps_b * np.ones((n, 1))
    # b_u = eps_b * np.ones((n, 1))
    b_l = b_sample - eps_b
    b_u = b_sample + eps_b
    bset = BoxSet(b, b_l, b_u)
    # bset = BoxSet
    # bset = ConstSet(b, 0.5 * np.ones((n, 1)))

    # obj = [ConvergenceResidual(x), ConvergenceResidual(y), ConvergenceResidual(z)]
    obj = [ConvergenceResidual(x), ConvergenceResidual(s)]
    # obj = OuterProdTrace(x)

    CP = VerificationProblem(N, [xset, yset, zset], [bset], obj, steps)

    # CP.print_cp()

    if solver_type == 'SDP':
        # return CP.solve(solver_type='SDP', add_RLT=add_RLT, verbose=verbose)
        return CP.canonicalize(solver_type='SDP', add_RLT=add_RLT, verbose=verbose), x
    if solver_type == 'GLOBAL':
        # return CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600, verbose=verbose)
        return CP.canonicalize(solver_type='GLOBAL', add_bounds=True,
                               TimeLimit=3600, verbose=verbose, model=gp_model), x


def get_b_sample(n, eps_b):
    return np.random.uniform(-eps_b, eps_b, n).reshape((n, 1))


def make_gurobi_model(timelimit=3600):
    m = gp.Model()
    m.setParam('NonConvex', 2)
    m.setParam('MIPGap', .1)
    m.setParam('TimeLimit', timelimit)
    return m


def mult_sample_expectation_global(n, m, N=2, b_samples=None, num_samples=1, eps_b=.01, r_x=1):
    timelimit = 3600
    model = make_gurobi_model(timelimit=timelimit)
    overall_obj = 0
    solvers = []
    solvers_x = []
    for i in range(num_samples):
        if b_samples is not None:
            b_sample = b_samples[i]
        else:
            b_sample = get_b_sample(n, eps_b)
        solver, x = OSQP_CP_noboxstack(n, m, N=N, eps_b=eps_b, r_x=r_x, b_sample=b_sample,
                                       solver_type='GLOBAL', gp_model=model)
        solvers.append(solver)
        solvers_x.append(x)
        overall_obj += solver.handler.objective
    if len(solvers) > 1:
        first_x0 = solvers[0].handler.get_iterate_var_map()[solvers_x[0]][0]
        for i in range(1, num_samples):
            handler_i = solvers[i].handler
            x0 = handler_i.get_iterate_var_map()[solvers_x[i]][0]
            # print(x0)
            model.addConstr(first_x0 == x0)
    model.setObjective(overall_obj / num_samples, gp.GRB.MAXIMIZE)
    model.optimize()
    return model.ObjVal, model.Runtime


def mult_sample_expectation_SDP(n, m, N=2, b_samples=None, num_samples=1, eps_b=.01, r_x=1, verbose=False):
    overall_obj = 0
    overall_constraints = []
    solvers = []
    solvers_x = []
    for i in range(num_samples):
        # b_sample = get_b_sample(n, eps_b)
        if b_samples is not None:
            b_sample = b_samples[i]
        else:
            b_sample = get_b_sample(n, eps_b)
        solver, x = OSQP_CP_noboxstack(n, m, N=N, eps_b=eps_b, r_x=r_x, b_sample=b_sample,
                                       solver_type='SDP', verbose=verbose)
        solvers.append(solver)
        solvers_x.append(x)
        # res = solver.solve()
        # print(res)
        overall_obj += solver.handler.sdp_obj
        overall_constraints += solver.handler.sdp_constraints
    if len(solvers) > 1:
        first_handler = solvers[0].handler.get_iteration_handler(0)
        first_x0_cp_var = first_handler.iterate_vars[solvers_x[0]].get_cp_var()
        for i in range(1, num_samples):
            handler_i = solvers[i].handler.get_iteration_handler(0)
            x0_cp_var = handler_i.iterate_vars[solvers_x[i]].get_cp_var()
            # print(x0_cp_var)
            overall_constraints += [first_x0_cp_var == x0_cp_var]
        # exit(0)
    prob = cp.Problem(cp.Maximize(overall_obj), overall_constraints)
    res = prob.solve(solver=cp.MOSEK, verbose=True)
    print(res / num_samples)
    time = prob.solver_stats.solve_time
    return res, time


def test_multiple_x_radius(n, m, save_dir, num_samples=1, max_N=5, verbose=False):
    radius_vals = [.01, .1, .5, 1, 5]
    # radius_vals = [.01]
    eps_b_val = .01
    N_vals = range(2, max_N+1)
    b_samples = []
    res_srlt_rows = []
    res_global_rows = []
    for _ in range(num_samples):
        b_samples.append(get_b_sample(n, eps_b_val))
    for N in N_vals:
        for r_x in radius_vals:
            print('N:', N, 'radius:', r_x)
            res_srlt, time_srlt = mult_sample_expectation_SDP(
                n, m, N=N, eps_b=eps_b_val, num_samples=num_samples, b_samples=b_samples, r_x=r_x, verbose=verbose)
            print(res_srlt, time_srlt)
            res_srlt_row = pd.Series(
                {
                    'num_iter': N,
                    'eps_b': eps_b_val,
                    'r_x': r_x,
                    'obj': res_srlt,
                    'solve_time': time_srlt,
                }
            )
            res_srlt_rows.append(res_srlt_row)
            res_global, time_global = mult_sample_expectation_global(
                n, m, N=N, eps_b=eps_b_val, num_samples=num_samples, b_samples=b_samples, r_x=r_x)
            print(res_global, time_global)
            res_global_row = pd.Series(
                {
                    'num_iter': N,
                    'eps_b': eps_b_val,
                    'r_x': r_x,
                    'obj': res_global,
                    'solve_time': time_global,
                }
            )
            res_global_rows.append(res_global_row)
    df_srlt = pd.DataFrame(res_srlt_rows)
    print(df_srlt)
    df_g = pd.DataFrame(res_global_rows)
    print(df_g)

    # exit(0)
    # s_fname = save_dir + 'sdp.csv'
    srlt_fname = save_dir + 'vary_radius_sdp_rlt.csv'
    g_fname = save_dir + 'vary_radius_global.csv'

    # df_s.to_csv(s_fname, index=False)
    df_srlt.to_csv(srlt_fname, index=False)
    df_g.to_csv(g_fname, index=False)


def main():
    np.random.seed(0)
    # N = 10
    m = 5
    n = 3
    max_N = 6
    # OSQP_CP_noboxstack(n, m, N=N)
    # solver_type = 'SDP'
    # solver_type = 'GLOBAL'
    # print(OSQP_CP_noboxstack(n, m, N=4, eps_b=.01, solver_type=solver_type, add_RLT=True, verbose=True))
    save_dir = \
        '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/OSQP/data/mult_eps_test/'
    # test_multiple_eps(n, m, save_dir, max_N=max_N, verbose=True)
    # mult_sample_expectation_SDP(n, m, num_samples=2, r_x=.1)
    # mult_sample_expectation_global(n, m, num_samples=2, r_x=.1)
    test_multiple_x_radius(n, m, save_dir, max_N=max_N)


if __name__ == '__main__':
    main()
