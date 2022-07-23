from joblib import Parallel, delayed
import numpy as np
import cvxpy as cp
import scipy.sparse as spa
import pandas as pd
from certification_problem.certification_problem import CertificationProblem
from certification_problem.variables.iterate import Iterate
from certification_problem.variables.parameter import Parameter
from certification_problem.high_level_alg_steps.hl_linear_step import HighLevelLinearStep
from certification_problem.basic_algorithm_steps.nonneg_orthant_proj_step import NonNegProjStep
import certification_problem.init_set as cpi
from certification_problem.init_set.centered_l2_ball_set import CenteredL2BallSet
from certification_problem.init_set.box_set import BoxSet
from certification_problem.init_set.ellipsoidal_set import EllipsoidalSet
from certification_problem.init_set.const_set import ConstSet
from certification_problem.objectives.convergence_residual import ConvergenceResidual

from PEPit import PEP
from PEPit.functions import SmoothStronglyConvexFunction, ConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


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
    CP = CertificationProblem(N, [xset], [bset], obj, steps)
    res = CP.solve(solver_type='GLOBAL')
    print(res)


def test_pep_part(A, t, out_fname=None, N_max=1):
    (m, n) = A.shape
    ATA = A.T @ A
    eigvals = np.linalg.eigvals(ATA)
    L = np.max(eigvals)
    mu = np.min(eigvals)
    print(L, mu)
    num_samples = 5000
    max_r = 0
    for _ in range(num_samples):
        test_x = sample_x(n)
        res, x = solve_single_NNLS_via_cvxpy(A, sample_b(m))
        r = np.linalg.norm(x - test_x)
        if r > max_r:
            max_r = r
    # print(max_r)
    df_rows = []
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
        df_rows.append(new_row)
    df = pd.DataFrame(df_rows)
    print(df)
    df.to_csv(out_fname, index=False)


def test_global_param_effect():
    np.random.seed(2)
    m = 10
    n = 5
    N = 6
    t = .05
    r = 1

    A = np.random.randn(m, n)
    # NNLS_cert_prob(n, m, spa.csc_matrix(A), N=N, t=t, r=r)
    print('--------WARM STARTING--------')
    sample_r = np.random.uniform(0, r)
    u = np.random.randn(m)
    u = u / np.linalg.norm(u)
    sample_b = u * sample_r
    # print(sample_b, np.linalg.norm(sample_b))
    _, center = solve_single_NNLS_via_cvxpy(A, sample_b)
    print('center:', center)
    # NNLS_cert_prob(n, m, spa.csc_matrix(A), N=N, t=t, r=r, center=center)

    # PEP part
    print('--------PEP BOUND--------')
    out_dir = '/Users/vranjan/Dropbox (Princeton)/ORFE/2022/algorithm-certification/experiments/NNLS/data/'
    test_pep_part(A, t, out_fname=out_dir + 'test_NNLS_multstep_PEP.csv', N_max=6)


def main():
    test_global_param_effect()


if __name__ == '__main__':
    main()
