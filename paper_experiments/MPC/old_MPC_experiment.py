# from quadcopter import QuadCopter
import numpy as np
import scipy.sparse as spa
from MPC_class import ModelPredictiveControl

from algoverify.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algoverify.basic_algorithm_steps.min_with_vec_step import MinWithVecStep
from algoverify.high_level_alg_steps.linear_step import LinearStep
from algoverify.init_set.box_set import BoxSet
from algoverify.init_set.l2_ball_set import L2BallSet
from algoverify.init_set.stack_set import StackSet
from algoverify.init_set.zero_set import ZeroSet
from algoverify.objectives.block_convergence_residual import BlockConvergenceResidual
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.variables.iterate import Iterate
from algoverify.variables.parameter import Parameter
from algoverify.verification_problem import VerificationProblem


def qp_cert_prob(orig_n, example, xinit_l, xinit_u, rho_const=True, K=1):
    print(xinit_l, xinit_u)
    P = example.qp_problem['P']
    A = example.qp_problem['A']
    print('P, A shape:', P.shape, A.shape)
    A.T @ A
    m, n = A.shape
    l = example.qp_problem['l']
    u = example.qp_problem['u']
    print(A.todense())
    print('l shape:', l.shape)
    exit(0)

    if rho_const:
        rho = np.eye(m)
    else:
        eq_idx = np.where(np.abs(u - l) <= 1e-5)
        # print(u-l)
        rho = np.ones(m)
        rho[eq_idx] *= 5
        rho = np.diag(rho)
    rho = spa.csc_matrix(rho)
    # rho_inv = np.linalg.inv(rho)
    rho_inv = spa.linalg.inv(rho)
    sigma = 1e-6

    l_noinit = l[orig_n:]
    u_noinit = u[orig_n:]
    print(l_noinit.shape, l_noinit, u_noinit)

    x_init = Parameter(orig_n, name='x_init')
    x_initset = BoxSet(x_init, -xinit_u.reshape(-1, 1), -xinit_l.reshape(-1, 1))
    # x_initset = BoxSet(x_init, xinit_l, xinit_u)
    l = Parameter(m, name='l')
    u = Parameter(m, name='u')
    # lset_rest = (l_noinit.reshape(-1, 1), l_noinit.reshape(-1, 1))
    # uset_rest = (u_noinit.reshape(-1, 1), u_noinit.reshape(-1, 1))
    lset_rest = (l_noinit, l_noinit)
    uset_rest = (u_noinit, u_noinit)
    # print(lset_rest, uset_rest)
    lset = StackSet(l, [x_init, lset_rest])
    uset = StackSet(u, [x_init, uset_rest])

    paramsets = [x_initset, lset, uset]

    x = Iterate(n, name='x')
    w = Iterate(m, name='w')
    z_tilde = Iterate(m, name='z_tilde')
    z = Iterate(m, name='z')
    y = Iterate(m, name='y')
    Iterate(m, name='y')

    In = np.eye(n)
    Im = np.eye(m)
    zeros_n = np.zeros((n, 1))
    zeros_m = np.zeros((m, 1))

    # step 1
    # (P + sigma I + A^T rho A) x = sigma x + A^T rho z - A^T y
    # print(P.shape, In.shape, rho.shape, A.shape)
    s1D = spa.csc_matrix(P + sigma * In + A.T @ rho @ A)
    s1A = spa.bmat([[sigma * In, A.T @ rho, -A.T]])
    s1b = zeros_n
    s1Dinv = spa.linalg.inv(s1D)
    step1 = LinearStep(x, [x, z, y], D=s1D, A=s1A, b=s1b, Dinv=s1Dinv)

    # step 2
    # w = Ax + rho^{-1} y
    s2D = spa.eye(m)
    s2A = spa.bmat([[A, rho_inv @ Im]])
    s2b = zeros_m
    s2Dinv = s2D
    step2 = LinearStep(w, [x, y], D=s2D, A=s2A, b=s2b, Dinv=s2Dinv)

    # step 3
    # z_tilde = max(w, l)
    step3 = MaxWithVecStep(z_tilde, w, l=l)
    # step3 = MaxWithVecStep(z, w, l=l)

    # step 4
    # z = min(z_tilde, u)
    step4 = MinWithVecStep(z, z_tilde, u=u)
    # step4 = LinearStep(z, [z_tilde], D=spa.eye(m), A=spa.eye(m), b=zeros_m, Dinv=spa.eye(m))

    # step 5
    # y = y + rho A x - rho z
    s5D = spa.eye(m)
    # s5A = spa.bmat([[Im, rho @ A, -rho @ Im]])
    s5A = spa.bmat([[Im, rho @ A, -rho]])
    s5b = zeros_m
    s5Dinv = s5D
    step5 = LinearStep(y, [y, x, z], D=s5D, A=s5A, b=s5b, Dinv=s5Dinv)

    # step 6
    # s = z + rho^{-1} y
    # s6D = spa.eye(m)
    s6A = spa.bmat([[Im, rho_inv]])
    # s6b = zeros_m
    # s6Dinv = s6D
    # step6 = LinearStep(s, [z, y], D=s6D, A=s6A, b=s6b, Dinv=s6Dinv)

    # steps = [step1, step2, step3, step5, step6]
    steps = [step1, step2, step3, step4, step5]

    x_c = np.zeros((n, 1))
    x_c[0] = -.5
    # xset = ZeroSet(x)
    xset = L2BallSet(x, x_c, 0)
    yset = ZeroSet(y)
    zset = ZeroSet(z)
    # sset = ZeroSet(s)

    initsets = [xset, yset, zset]
    # initsets = [xset, yset, zset, sset]

    # obj = [ConvergenceResidual(x), ConvergenceResidual(s)]
    obj = [ConvergenceResidual(x), BlockConvergenceResidual([z, y], s6A)]

    CP = VerificationProblem(K, initsets, paramsets, obj, steps)

    out_g, out_time = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600)
    print(out_g, out_time)

    VerificationProblem(K, initsets, paramsets, obj, steps)
    # out_s = CP2.solve(solver_type='SDP_CUSTOM')


def qp_cert_prob_max_only(orig_n, example, xinit_l, xinit_u, rho_const=True, K=1):
    print(xinit_l, xinit_u)
    P = example.qp_problem['P']
    A = example.qp_problem['A']
    print('P, A shape:', P.shape, A.shape)
    A.T @ A
    m, n = A.shape
    l = example.qp_problem['l']
    # u = example.qp_problem['u']

    lbox_lower = l.copy()
    lbox_upper = l.copy()

    # lbox_lower = l
    # lbox_upper = l
    # lbox_lower[:xinit_l.shape[0]] = xinit_l
    # lbox_upper[:xinit_u.shape[0]] = xinit_u

    lbox_lower[:xinit_l.shape[0]] = -xinit_u
    lbox_upper[:xinit_u.shape[0]] = -xinit_l

    l_param = Parameter(m, name='l')
    lset = BoxSet(l_param, lbox_lower.reshape(-1, 1), lbox_upper.reshape(-1, 1))

    paramsets = [lset]

    x = Iterate(n, name='x')
    w = Iterate(m, name='w')
    z = Iterate(m, name='z')
    y = Iterate(m, name='y')
    s = Iterate(m, name='s')

    xset = ZeroSet(x)
    yset = ZeroSet(y)
    zset = ZeroSet(z)

    initsets = [xset, yset, zset]

    In = spa.eye(n)
    Im = spa.eye(m)
    rho = np.eye(m)
    rho_inv = np.linalg.inv(rho)
    sigma = 1e-6

    print('check lset not zero:', np.linalg.norm(lset.u - lset.l))

    # step 1
    # Solve (P + sigma In + rho A^TA)x^{k+1} = sigma x^k + rho A^T z^k - A^T y^k
    s1D = spa.csc_matrix(P + sigma * In + A.T @ rho @ A)
    s1Dinv = spa.linalg.inv(s1D)
    s1A = spa.bmat([[sigma * In, A.T @ rho, -A.T]])
    s1b = np.zeros((n, 1))
    step1 = LinearStep(x, [x, z, y], D=s1D, A=s1A, b=s1b, Dinv=s1Dinv)

    # step 2
    # w^{k+1} = A x^{k+1} + rho^{-1} y^k
    s2D = spa.eye(m)
    s2Dinv = s2D
    s2A = spa.bmat([[A, rho_inv]])
    s2b = np.zeros((m, 1))
    step2 = LinearStep(w, [x, y], D=s2D, A=s2A, b=s2b, Dinv=s2Dinv)

    # step 3
    # z^{k+1} = Pi(w^{k+1})
    step3 = MaxWithVecStep(z, w, l=l_param)

    # step 4
    # y^{k+1} = y^k + rho @ A x^{k+1} - rho @ z^{k+1}
    s4D = spa.eye(m)
    s4Dinv = s4D
    # print(Im.shape, (rho @ A).shape, (-rho).shape)
    s4A = spa.bmat([[Im, rho @ A, -rho]])
    np.zeros((m, 1))
    step4 = LinearStep(y, [y, x, z], D=s4D, A=s4A, b=s2b, Dinv=s4Dinv)

    # step 5
    # s^{k+1} = z^{k+1} + rho^{-1} y^{k+1}
    s5D = spa.eye(m)
    s5Dinv = s5D
    s5A = spa.bmat([[Im, rho_inv]])
    s5b = np.zeros((m, 1))
    step5 = LinearStep(s, [z, y], D=s5D, A=s5A, b=s5b, Dinv=s5Dinv)

    steps = [step1, step2, step3, step4, step5]

    obj = [ConvergenceResidual(x), ConvergenceResidual(s)]

    CP = VerificationProblem(K, initsets, paramsets, obj, steps)

    out_g, out_time = CP.solve(solver_type='GLOBAL', add_bounds=True, TimeLimit=3600)
    print(out_g, out_time)
    iter_var_map = CP.solver.handler.get_iterate_var_map()
    param_var_map = CP.solver.handler.get_param_var_map()

    print(param_var_map[l_param].X)
    # print(iter_var_map[x][-1].X)
    print(iter_var_map[z][-2].X)


def main():
    n = 4
    T = 100
    mpc = ModelPredictiveControl(n=n, T=T, seed=0)
    mpc.qp_problem['P']
    mpc.qp_problem['A']
    mpc.qp_problem['l']
    mpc.qp_problem['u']
    # print(mpc.x0)
    # exit(0)
    num_sim = 50
    x_inits = []
    np.random.seed(0)
    for i in range(num_sim):
        # self.cvxpy_problem, self.cvxpy_variables, self.cvxpy_param
        print('----')
        x_inits.append(mpc.x0)
        cp_prob, cp_vars, _ = mpc.cvxpy_problem, mpc.cvxpy_variables, mpc.cvxpy_param
        # x, u = cp_vars
        x, u = cp_vars
        res = cp_prob.solve()
        print(res)
        print(u.value, u.shape)
        print(np.round(x.value, 4), x.shape)
        # exit(0)

        x0_ref = np.zeros(n)
        x0_ref[0] = -.5

        x0_new = x.value[:, -1] + np.random.normal(scale=.001, size=(n,))
        x0_ref = x0_ref + x0_new
        print(x0_new)
        print(x0_ref)
        # exit(0)
        # print(x0_new, x0_ref)
        mpc.update_x0(x0_ref)
    # print(x_inits)
    # print(np.array(x_inits))
    X = np.array(x_inits)
    box_l = np.zeros(X.shape[1])
    box_u = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        box_l[i] = np.min(X[:, i])
        box_u[i] = np.max(X[:, i])
    print(box_l, box_u, box_l.shape)
    print(mpc.qp_problem['l'].shape)
    exit(0)
    # print(box_l, box_u)
    # print(P.shape, A.shape)
    # print(l, u, u - l)
    # print(np.linalg.eigvals(P.todense()))
    qp_cert_prob(n, mpc, box_l, box_u, rho_const=True, K=5)
    # qp_cert_prob_max_only(n, mpc, box_l, box_u, K=2)
    # print(mpc.qp_problem['q'])


if __name__ == '__main__':
    main()
