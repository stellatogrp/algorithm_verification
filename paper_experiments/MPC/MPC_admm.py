import numpy as np
from MPC_class import ModelPredictiveControl


def run_admm_single(P, A, l, u, K=300):

    # rho = 1
    # rho_inv = 1 / rho

    sigma = 1e-6
    m, n = A.shape
    # ATA = A.T @ A
    In = np.eye(n)
    # Im = np.eye(m)
    rho = np.eye(m)
    rho_inv = np.linalg.inv(rho)

    def Pi(x):
        # return np.minimum(u, np.maximum(x, l))
        return np.maximum(x, l)

    xk = np.zeros(n)
    xk[0] = -.5
    np.zeros(m)
    yk = np.zeros(m)
    zk = np.zeros(m)
    sk = zk + rho_inv @ yk
    x_iterates = [xk]
    y_iterates = [yk]
    z_iterates = [zk]
    s_iterates = [sk]

    lhs = P + sigma * In + A.T @ rho @ A
    for _ in range(K):
        rhs = sigma * xk + A.T @ (rho @ zk - yk)
        # rhs_mat = np.bmat([[sigma * np.eye(n), A.T @ rho, -A.T.todense()]])
        # rhs_stack = np.hstack([xk, zk, yk]).reshape((-1, 1))
        # # rhs_vec = rhs_mat @ rhs_stack
        # rhs_vec = rhs_mat.dot(rhs_stack)
        # print(rhs_mat.shape, rhs_stack.shape, rhs_vec.shape)
        # rhs_vec = np.reshape(rhs_vec, (-1, ))
        # print(rhs_vec)
        # exit(0)
        # print(lhs.shape, rhs.shape, rhs)
        # xkplus1 = np.linalg.solve(lhs, rhs)
        # print(rhs_mat.shape, rhs_vec.shape)
        xkplus1 = np.linalg.solve(lhs, rhs)
        wkplus1 = A @ xkplus1 + rho_inv @ yk
        zkplus1 = Pi(wkplus1)
        ykplus1 = yk + rho @ (A @ xkplus1 - zkplus1)
        skplus1 = zkplus1 + rho_inv @ ykplus1
        # print('x:', xkplus1)
        # print('w:', wkplus1)
        # print('z:', zkplus1)
        # print('y:', ykplus1)
        # print('s:', skplus1)

        x_iterates.append(xkplus1)
        y_iterates.append(ykplus1)
        z_iterates.append(zkplus1)
        s_iterates.append(skplus1)

        xk = xkplus1
        yk = ykplus1
        zk = zkplus1
        sk = skplus1
        # print(xk.shape, yk.shape, zk.shape)
    fixed_point_resids = []
    for i in range(1, K+1):
        x_resid = np.linalg.norm(x_iterates[i] - x_iterates[i-1]) ** 2
        s_resid = np.linalg.norm(s_iterates[i] - s_iterates[i-1]) ** 2
        fixed_point_resids.append(x_resid + s_resid)
    # print('tested for:', np.round(l, 3))
    # print(np.round(fixed_point_resids, 3))
    # print('direct admm x:', np.round(x_iterates, 4))
    # print('direct admm y:', np.round(y_iterates, 3))
    # print('direct admm z:', np.round(z_iterates, 3))
    return x_iterates, y_iterates, z_iterates, fixed_point_resids


def set_up_admm_run(mpc, x_inits, K=100):
    N = x_inits.shape[0]
    print(N)

    resids = []
    for i in range(N):
        x0 = x_inits[i, :]
        print(f'----{x0}----')
        mpc.update_x0(x0)
        P = mpc.qp_problem['P']
        A = mpc.qp_problem['A']
        l = mpc.qp_problem['l']
        u = mpc.qp_problem['u']
        # print(l, u)
        x_iterates, cp_var, _, fixed_point_resids = run_admm_single(P, A, l, u, K=1)
        print('fixed pt:', fixed_point_resids[-1])

        cp_prob, cp_vars = mpc.cvxpy_problem, mpc.cvxpy_variables
        cp_prob.solve()
        # print(res)
        x_admm = x_iterates[-1]
        # print(.5 * x_admm.T @ P @ x_admm)
        print(cp_vars[0].value)
        print('final x:', np.round(x_admm, 4))
        resids.append(fixed_point_resids[-1])
    print(np.max(resids))


def main():
    n = 2
    mpc = ModelPredictiveControl(n=n, T=2)
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
        x_inits.append(mpc.x0)
        cp_prob, cp_vars, _ = mpc.cvxpy_problem, mpc.cvxpy_variables, mpc.cvxpy_param
        # x, u = cp_vars
        x, _ = cp_vars
        cp_prob.solve()
        # print(res)
        # print(u.value, u.shape)
        # print(np.round(x.value, 4), x.shape)

        x0_ref = np.zeros(n)
        x0_ref[0] = -.5

        x0_new = x.value[:, -1] + np.random.normal(scale=.01, size=(n,))
        x0_ref = x0_ref + x0_new
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
    print(box_l, box_u)
    print(X[0, :])

    set_up_admm_run(mpc, X)


if __name__ == '__main__':
    main()
