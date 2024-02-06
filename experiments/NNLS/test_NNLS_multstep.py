import cvxpy as cp
import numpy as np
import pandas as pd
import scipy.sparse as spa
from PEPit import PEP
from PEPit.functions import ConvexFunction, SmoothStronglyConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step
from tqdm import trange

from algoverify.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
from algoverify.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from algoverify.init_set.box_set import BoxSet
from algoverify.init_set.centered_l2_ball_set import CenteredL2BallSet
from algoverify.init_set.const_set import ConstSet
from algoverify.init_set.ellipsoidal_set import EllipsoidalSet
from algoverify.objectives.convergence_residual import ConvergenceResidual
from algoverify.variables.iterate import Iterate
from algoverify.variables.parameter import Parameter
from algoverify.verification_problem import VerificationProblem


def test_PEPit_val(L, mu, t, r, N=1):
    problem = PEP()
    verbose = 0

    # Declare a strongly convex smooth function and a closed convex proper function
    f1 = problem.declare_function(SmoothStronglyConvexFunction, mu=mu, L=L)
    f2 = problem.declare_function(ConvexFunction)
    func = f1 + f2

    # Start by defining its unique optimal point xs = x_*
    xs = func.stationary_point()

    # Then define the starting point x0 of the algorithm
    x0 = problem.set_initial_point()

    # Set the initial constraint that is the distance between x0 and x^*
    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)

    # Run the proximal gradient method starting from x0
    x = x0
    x_vals = [x0]
    for _ in range(N):
        y = x - t * f1.gradient(x)
        x, _, _ = proximal_step(y, f2, t)
        x_vals.append(x)

    # Set the performance metric to the distance between x and xs
    problem.set_performance_metric((x_vals[-1] - x_vals[-2]) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Return the worst-case guarantee of the evaluated method ( and the reference theoretical value)
    return pepit_tau


def run_single_prox_grad_descent(tATA, tAtb, num_iter=5, x_init=None):
    def nonneg_proj(x):
        return np.maximum(x, 0)
    n = tATA.shape[0]
    In = np.eye(n)
    C = In - tATA
    if x_init is None:
        x_init = np.zeros(n)
    iterates = [x_init]
    xk = x_init
    for i in range(num_iter):
        xk = nonneg_proj(C @ xk + tAtb)
        iterates.append(xk)

    # conv_resid = np.linalg.norm(iterates[-1] - iterates[-2])
    return iterates


def get_centered_l2_xset(*args):
    x = args[0]
    r = args[1]
    return CenteredL2BallSet(x, r=r)


def sample_x(n):
    r = np.random.uniform()
    rand_dir = np.random.randn(n)
    rand_dir_unit = rand_dir / np.linalg.norm(rand_dir)
    return rand_dir_unit * r


def get_off_center_l2_xset(*args):
    x = args[0]
    r = args[1]
    c = args[2]
    n = x.get_dim()
    Q = (1 / r ** 2) * np.eye(n)
    return EllipsoidalSet(x, Q, c)


def get_bset(b):
    m = b.get_dim()
    b_l = 1 * np.ones(m)
    b_u = 3 * np.ones(m)
    return BoxSet(b, b_l, b_u)


def sample_b(m):
    unit_sample = np.random.rand(m)
    return 1 + 2 * unit_sample


def solve_single_NNLS_via_cvxpy(A, b):
    n = A.shape[1]
    x = cp.Variable(n)
    obj = .5 * cp.sum_squares(A @ x - b)
    constraints = [x >= 0]
    problem = cp.Problem(cp.Minimize(obj), constraints)
    res = problem.solve()
    return res, x.value


def NNLS_cert_prob(n, m, A, N=1, t=.05, r=1, center=None):
    '''
        Set up and solve certification problem for:
        min (1/2) || Ax - b ||_2^2 s.t. x >= 0
        via proximal gradient descent
    :param n: dimension of x
    :param m: dimension of b
    :param A:
    :param t: step size
    :return:
    '''
    y = Iterate(n, name='y')
    x = Iterate(n, name='x')
    b = Parameter(m, name='b')
    # print(y, x, b)

    ATA = A.T @ A
    In = spa.eye(n)
    C = spa.bmat([[In - t * ATA, t * A.T]])
    zeros = np.zeros(n)

    # xset = bset_func(x, r)
    # xset = get_off_center_l2_xset(x, r, np.ones(n))
    if center is None:
        xset = get_centered_l2_xset(x, r)
    else:
        # xset = get_off_center_l2_xset(x, .1 * r, center)
        xset = ConstSet(x, center)

    # bset = CenteredL2BallSet(b, r=r)
    bset = get_bset(b)

    step1 = HighLevelLinearStep(y, [x, b], D=In, A=C, b=zeros)
    step2 = NonNegProjStep(x, y)

    steps = [step1, step2]

    obj = ConvergenceResidual(x)
    CP = VerificationProblem(N, [xset], [bset], obj, steps)
    res = CP.solve(solver_type='GLOBAL')
    print(res)


def test_pep_part(A, t, ws_center, out_fname_avg=None, out_fname_pep=None, N_max=1):
    (m, n) = A.shape
    ATA = A.T @ A
    tATA = t * ATA
    eigvals = np.linalg.eigvals(ATA)
    L = np.max(eigvals)
    mu = np.min(eigvals)
    print(L, mu)
    num_samples = 100
    max_r = 0
    iterate_rows = []
    #  ws_iterate_rows = []
    for _ in trange(num_samples):
        test_x = sample_x(n)
        test_b = sample_b(m)
        res, x = solve_single_NNLS_via_cvxpy(A, test_b)
        r = np.linalg.norm(x - test_x)
        if r > max_r:
            max_r = r

        tATb = t * A.T @ test_b
        iterates = run_single_prox_grad_descent(tATA, tATb, num_iter=N_max, x_init=test_x)
        ws_iterates = run_single_prox_grad_descent(tATA, tATb, num_iter=N_max, x_init=ws_center)
        for k in range(1, N_max+1):
            res = np.linalg.norm(iterates[k] - iterates[k-1] ** 2)
            ws_res = np.linalg.norm(ws_iterates[k] - ws_iterates[k-1] ** 2)
            iter_row = pd.Series(
                {
                    'max_N': k,
                    'res': res,
                    'ws_res': ws_res,
                }
            )
            iterate_rows.append(iter_row)
        # tATb = t * A.T @ test_b
    # print(max_r)
    df_avg = pd.DataFrame(iterate_rows)
    print(df_avg)
    df_avg.to_csv(out_fname_avg, index=False)
    # exit(0)

    df_pep_rows = []
    for N in range(1, N_max+1):
        pep_bound = test_PEPit_val(L, mu, t, max_r, N=N)
        print(N, pep_bound)
        new_row = pd.Series(
            {
                'N': N,
                'n': n,
                'm': m,
                'pep_bound': pep_bound,
            }
        )
        df_pep_rows.append(new_row)
    df_pep = pd.DataFrame(df_pep_rows)
    print(df_pep)
    df_pep.to_csv(out_fname_pep, index=False)


def test_global_param_effect():
    np.random.seed(2)
    m = 10
    n = 5
    #  N = 6
    t = .05
    r = 1

    A = np.random.randn(m, n)
    # NNLS_cert_prob(n, m, spa.csc_matrix(A), N=N, t=t, r=r)
    print('--------WARM STARTING--------')
    sample_r = np.random.uniform(0, r)
    u = np.random.randn(m)
    u = u / np.linalg.norm(u)
    sample_b_val = u * sample_r
    # print(sample_b, np.linalg.norm(sample_b))
    _, center = solve_single_NNLS_via_cvxpy(A, sample_b_val)
    print('center:', center)
    # NNLS_cert_prob(n, m, spa.csc_matrix(A), N=N, t=t, r=r, center=center)

    # PEP part
    print('--------PEP BOUND--------')
    #  out_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/NNLS/data/'
    tATA = t * A.T @ A
    print('testing')
    num_samples = 10000
    cs_vals = []
    ws_vals = []
    prox_grad_iter = 6
    print('prox grad iter', prox_grad_iter)

    for _ in trange(num_samples):
        #  test_b = sample_b(m)
        tATb = t * A.T @ sample_b(m)
        test_x = sample_x(n)
        cs_test = run_single_prox_grad_descent(tATA, tATb, num_iter=prox_grad_iter, x_init=test_x)
        ws_test = run_single_prox_grad_descent(tATA, tATb, num_iter=prox_grad_iter, x_init=center)
        # print('test', np.linalg.norm(test[-1] - test[-2]) ** 2)
        cs_vals.append(np.linalg.norm(cs_test[-1] - cs_test[-2]) ** 2)
        ws_vals.append(np.linalg.norm(ws_test[-1] - ws_test[-2]) ** 2)
    print('cold starting', np.mean(cs_vals))
    print('warm starting', np.mean(ws_vals))
    # test_pep_part(A, t, center, out_fname_avg=out_dir +
    #               'test_NNLS_multstep_avg.csv',
    #               out_fname_pep=out_dir +
    #               'test_NNLS_multstep_PEP.csv', N_max=6)


def main():
    test_global_param_effect()


if __name__ == '__main__':
    main()
