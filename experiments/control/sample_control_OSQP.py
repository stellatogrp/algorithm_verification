import cvxpy as cp
import numpy as np
import pandas as pd
from control_example import ControlExample
from PEPit import PEP
from PEPit.functions import ConvexFunction, SmoothStronglyConvexFunction
from PEPit.primitive_steps.proximal_step import proximal_step


def generate_problem_data(n):
    return ControlExample(n)


def sample_xinit(example):
    xmin = example.xmin
    xmax = example.xmax
    return np.random.uniform(xmin, xmax)


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


def get_ws_vals(example):
    A = example.qp_problem['A']
    _, full_n = A.shape
    P = example.qp_problem['P']
    l = example.qp_problem['l']
    u = example.qp_problem['u']

    x_val = solve_with_cvxpy(P, A, l, u)
    z_val = A @ x_val

    return x_val, z_val


def solve_with_cvxpy(P, A, l, u):
    _, full_n = A.shape
    x_var = cp.Variable(full_n)
    obj = .5 * cp.quad_form(x_var, P)
    constraints = [l <= A @ x_var, A @ x_var <= u]
    problem = cp.Problem(cp.Minimize(obj), constraints)
    problem.solve()
    return x_var.value


def run_admm(P, A, l, u, x0=None, z0=None, N=300):
    rho = 1
    rho_inv = 1 / rho
    sigma = 1
    m, n = A.shape
    ATA = A.T @ A
    In = np.eye(n)
    # Im = np.eye(m)

    def Pi(x):
        return np.minimum(u, np.maximum(x, l))

    if x0 is not None:
        xk = x0
    else:
        xk = np.zeros(n)
    yk = np.zeros(m)
    if z0 is not None:
        zk = z0
    else:
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
    return x_iterates, s_iterates, fixed_point_resids


def run_avg_exp():
    n = 2
    max_N = 9
    example = generate_problem_data(n)
    P = example.qp_problem['P']
    A = example.qp_problem['A']
    l = example.qp_problem['l'].copy()
    u = example.qp_problem['u'].copy()
    num_samples = 1000
    x_ws, z_ws = get_ws_vals(example)
    # print(x_ws, z_ws)
    cs = []
    ws = []
    max_r = 0
    for _ in range(num_samples):
        xinit_sample = sample_xinit(example)
        l[:n] = -xinit_sample
        u[:n] = -xinit_sample
        x_opt = solve_with_cvxpy(P, A, l, u)
        # print(x_opt)
        if x_opt is not None:
            r = np.linalg.norm(x_opt)
            # print(r)
            if r > max_r:
                max_r = r

        x_ws, z_ws = get_ws_vals(example)
        _, _, cs_resids = run_admm(P, A, l, u, N=max_N)
        _, _, ws_resids = run_admm(P, A, l, u, x0=x_ws, z0=z_ws, N=max_N)
        # print(cs_resids, ws_resids)
        cs.append(cs_resids)
        ws.append(ws_resids)
    print(max_r)
    csv_rows = []
    for i in range(max_N):
        csi = [arr[i] for arr in cs]
        wsi = [arr[i] for arr in ws]
        # print(np.mean(csi), np.mean(wsi))
        cs_resid_avg = np.mean(csi)
        ws_resid_avg = np.mean(wsi)
        row = pd.Series(
            {
                'num_iter': i+1,
                'cs_resid_avg': cs_resid_avg,
                'ws_resid_avg': ws_resid_avg,
            }
        )
        csv_rows.append(row)
    df = pd.DataFrame(csv_rows)
    print(df)


def test_admm_pep(L, mu, alpha, theta, r=1, N=1):
    verbose = 0
    # Instantiate PEP
    problem = PEP()

    # Declare a convex and a smooth convex function.
    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    # Define the function to optimize as the sum of func1 and func2
    func = func1 + func2

    # Start by defining its unique optimal point xs = x_* and its function value fs = F(x_*)
    xs = func.stationary_point()
    # fs = func(xs)

    # Then define the starting point x0 of the algorithm and its function value f0
    x0 = problem.set_initial_point()

    # Compute n steps of the Douglas-Rachford splitting starting from x0
    x = [x0 for _ in range(N)]
    w = [x0 for _ in range(N + 1)]
    for i in range(N):
        x[i], _, _ = proximal_step(w[i], func2, alpha)
        y, _, fy = proximal_step(2 * x[i] - w[i], func1, alpha)
        w[i + 1] = w[i] + theta * (y - x[i])

    # Set the initial constraint that is the distance between x0 and xs = x_*
    problem.set_initial_condition((x[0] - xs) ** 2 <= r ** 2)

    # Set the performance metric to the final distance to the optimum in function values
    # problem.set_performance_metric((func2(y) + fy) - fs)
    problem.set_performance_metric((x[-1] - x[-2]) ** 2)

    # Solve the PEP
    pepit_verbose = max(verbose, 0)
    pepit_tau = problem.solve(verbose=pepit_verbose)

    # Return the worst-case guarantee of the evaluated method (and the upper theoretical value)
    return pepit_tau


def test_pep():
    n = 2
    # max_N = 9
    example = generate_problem_data(n)
    P = example.qp_problem['P']
    # A = example.qp_problem['A']
    # l = example.qp_problem['l'].copy()
    # u = example.qp_problem['u'].copy()

    eigvals = np.linalg.eigvals(P.todense())
    L, mu = np.max(eigvals), np.min(eigvals)
    print(L, mu)
    print(test_admm_pep(L, mu, 1, 1, r=24, N=2))


def main():
    # run_avg_exp()
    test_pep()


if __name__ == '__main__':
    main()
