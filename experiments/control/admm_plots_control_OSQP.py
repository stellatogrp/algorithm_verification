import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from control_example import ControlExample

# import scipy.sparse as spa

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",   # This is needed only in the slides
    "font.sans-serif": ["Helvetica Neue"],   # This is needed only in the slides
    "font.size": 12,   # In the paper you can put 11 or 12
})


def generate_problem(n):
    return ControlExample(n)


def run_admm(P, A, l, u, N):
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
    print(np.round(fixed_point_resids, 3))
    # print('direct admm x:', np.round(x_iterates, 4))
    # print('direct admm y:', np.round(y_iterates, 3))
    # print('direct admm z:', np.round(z_iterates, 3))
    return fixed_point_resids


def x_init_str_to_np_array(x):
    test = x[1: -1].split()
    return np.round(np.array([float(val) for val in test]), 3)


def make_admm_plots(df, n):
    example = generate_problem(n)
    P = example.qp_problem['P']
    A = example.qp_problem['A']
    # ATA = A.T @ A
    full_m, full_n = A.shape
    l = example.qp_problem['l']
    u = example.qp_problem['u']
    l_test = l.copy()
    u_test = u.copy()
    # xmin = example.xmin
    # xmax = example.xmax
    # print(xmin, xmax)
    # print(np.linalg.eigxvals(ATA.todense()))

    print(df)
    N_vals = df['num_iter'].to_numpy()
    max_N = max(N_vals)
    # max_N = 2
    x_init_vals = df['x_init'].to_numpy()
    val_N_dict = {}
    for i, val in enumerate(x_init_vals):
        arr = tuple(x_init_str_to_np_array(val))
        # print(arr)
        if arr not in val_N_dict:
            val_N_dict[arr] = []
        val_N_dict[arr].append(i+1)
    # print(l_test, u_test)
    for key in val_N_dict:
        l_test[:n] = key
        u_test[:n] = key
        # print('l, u:', l_test, u_test)
        out = run_admm(P, A, l_test, u_test, max_N)
        # break
        print(out)


def main():
    data_dir = '/home/vranjan/algorithm-certification/experiments/control/data/'
    fname = data_dir + 'test_xinitn2N10.csv'
    df = pd.read_csv(fname)
    n = 2
    make_admm_plots(df, n)


if __name__ == '__main__':
    main()
