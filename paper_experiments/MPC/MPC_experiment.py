# from quadcopter import QuadCopter
import numpy as np
from MPC_class import ModelPredictiveControl


def main():
    n = 2
    mpc = ModelPredictiveControl(n=n, T=2)
    P = mpc.qp_problem['P']
    A = mpc.qp_problem['A']
    l = mpc.qp_problem['l']
    u = mpc.qp_problem['u']
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
    print(np.array(x_inits))
    X = np.array(x_inits)
    box_l = np.zeros(X.shape[1])
    box_u = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        box_l[i] = np.min(X[:, i])
        box_u[i] = np.max(X[:, i])
    print(box_l, box_u)
    print(P.shape, A.shape)
    print(l, u, u - l)
    print(np.linalg.eigvals(P.todense()))


if __name__ == '__main__':
    main()
