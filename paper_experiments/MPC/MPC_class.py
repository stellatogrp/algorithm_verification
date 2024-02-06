import cvxpy
import numpy as np
import scipy.linalg as sla
import scipy.sparse as spa


class ModelPredictiveControl(object):
    '''
    Control QP example
    '''

    def __init__(self, n=6, T=5, seed=1):
        '''
        Generate problem in QP format and CVXPY format
        '''
        # Set random seed
        np.random.seed(seed)

        # Generate random dynamics
        self.nx = int(n)       # States
        self.nu = int(n / 2)   # Inputs

        self.A = spa.eye(self.nx) + .1 * spa.random(self.nx, self.nx,
                                                    density=1.0,
                                                    data_rvs=np.random.randn)

        # Restrict eigenvalues of A to be less than 1
        lambda_values, V = np.linalg.eig(self.A.todense())
        abs_lambda_values = np.abs(lambda_values)

        # Enforce eigenvalues to be maximum norm 1
        for i in range(len(lambda_values)):
            lambda_values[i] = lambda_values[i] \
                if abs_lambda_values[i] < 1 - 1e-02 else \
                lambda_values[i] / (abs_lambda_values[i] + 1e-02)

        # Reconstruct A = V * Lambda * V^{-1}
        self.A = spa.csc_matrix(
            V.dot(np.diag(lambda_values)).dot(np.linalg.inv(V)).real
        )

        self.B = spa.random(self.nx, self.nu, density=1.0,
                            data_rvs=np.random.randn)

        # Control penalty
        self.R = .1 * spa.eye(self.nu)
        ind07 = np.random.rand(self.nx) < 1.0   # Random 30% data
        # Choose only 30% of nonzero elements
        diagQ = np.multiply(np.random.rand(self.nx), ind07)
        self.Q = spa.diags(diagQ)
        QN = sla.solve_discrete_are(self.A.todense(), self.B.todense(),
                                    self.Q.todense(), self.R.todense())
        self.QN = spa.csc_matrix(QN.dot(QN.T))

        # Control diff penalty (used to penalize u_{k+1} - u_k)
        # self.CR = .01 * spa.eye(self.nu)
        self.CR = 0 * spa.eye(self.nu)

        # # Constants for angle penalty:
        # self.m = 1
        # self.g = 9.8
        # self.mg = self.m * self.g
        # self.gamma = 1
        # self.gamma_mg = self.gamma * self.mg
        # self.N_hs = 4
        # self.gmg_ones = np.tile(self.gamma_mg, self.N_hs)
        # self.c_arr = []
        # for j in range(self.N_hs):
        #     c = np.array([np.cos(j*2*np.pi/self.N_hs), np.sin(j*2*np.pi/self.N_hs), -self.gamma])
        #     self.c_arr.append(c)
        # self.C_hs = np.vstack(self.c_arr)
        # # print(self.C_hs, self.gmg_ones)
        # # exit(0)

        # # self.QN = spa.csc_matrix(QN.dot(QN))  # Ensure symmetric PSD
        # # self.QN = 10 * self.Q

        # Input ad state bounds
        self.umin = - 1.0 * np.random.rand(self.nu)
        self.umax = -self.umin
        self.umin = -2
        self.umax = 2
        # self.xmin = -1.0 - np.random.rand(self.nx)
        # self.xmax = -self.xmin
        # self.xmin = -.1 * np.ones(n)
        # self.xmax = .1 * np.ones(n)
        self.xmin = - 1.1 * np.ones(n)
        self.xmax = 1.1 * np.ones(n)

        # Initial state (constrain to be within lower and upper bound)
        # self.x0 = np.random.rand(self.nx)
        # min_x0 = .5 * self.xmin
        # max_x0 = .5 * self.xmax

        # for i in range(self.nx):
        #    self.x0[i] = min_x0[i] + \
        #        self.x0[i] * (max_x0[i] - min_x0[i])

        self.x0 = np.zeros(self.nx)
        self.x0[0] = -.5
        min_x0 = np.zeros(self.nx)
        min_x0[0] = -1.5
        max_x0 = np.zeros(self.nx)
        max_x0[0] = -.5

        self.min_x0 = min_x0
        self.max_x0 = max_x0

        # Horizon length
        self.T = T

        self.qp_problem = self._generate_qp_problem()
        self.cvxpy_problem, self.cvxpy_variables, self.cvxpy_param = \
            self._generate_cvxpy_problem()

    @staticmethod
    def name():
        return 'Control'

    def _generate_qp_problem(self):
        '''
        Generate QP problem
        '''

        # Get input-state dimensions
        (nx, nu) = self.B.shape

        # Objective
        Px = spa.kron(spa.eye(self.T), self.Q)
        Pu = spa.kron(spa.eye(self.T), self.R + 2 * self.CR)
        off_diags = spa.eye(self.T, k=1) + spa.eye(self.T, k=-1)
        CR_reg = spa.kron(off_diags, -self.CR)
        Pu = Pu + CR_reg
        P = 2. * spa.block_diag([Px, self.QN, Pu]).tocsc()
        q = np.zeros((self.T + 1) * nx + self.T * nu)

        # Dynamics
        Ax = spa.kron(spa.eye(self.T + 1), -spa.eye(nx)) + \
            spa.kron(spa.eye(self.T + 1, k=-1), self.A)
        Au = spa.kron(spa.vstack([spa.csc_matrix((1, self.T)),
                                  spa.eye(self.T)]),
                      self.B)
        A = spa.hstack([Ax, Au])
        l = self._b(self.x0)
        u = self._b(self.x0)

        # Constraints without bounds
        A_nobounds = A.copy()
        l_nobounds = np.copy(l)
        u_nobounds = np.copy(u)

        # Initialize separate bounds variables
        lx = np.array([])
        ux = np.array([])

        # State constraints
        l = np.append(l, np.tile(self.xmin, self.T + 1))
        u = np.append(u, np.tile(self.xmax, self.T + 1))
        A = spa.vstack([A,
                        spa.hstack([spa.eye((self.T + 1)*nx),
                                    spa.csc_matrix(((self.T + 1)*nx,
                                                    self.T * nu))])
                        ]).tocsc()
        lx = np.append(lx, np.tile(self.xmin, self.T + 1))
        ux = np.append(ux, np.tile(self.xmax, self.T + 1))

        # Input constraints
        l = np.append(l, np.tile(self.umin, self.T))
        u = np.append(u, np.tile(self.umax, self.T))
        A = spa.vstack([A,
                        spa.hstack([spa.csc_matrix((self.T*nu, (self.T+1)*nx)),
                                    spa.eye(self.T*nu)])
                        ]).tocsc()
        lx = np.append(lx, np.tile(self.umin, self.T))
        ux = np.append(ux, np.tile(self.umax, self.T))

        # Angle constraints
        spa.eye(self.T)
        # temp = spa.kron(I, self.C_hs)

        # Z = spa.csc_matrix(np.zeros((temp.shape[0], (self.T + 1) * nx)))
        # print(Z.shape, temp.shape)
        # print(A.shape)
        # self.ang_cons = spa.hstack([Z, temp])
        # print(self.ang_cons.todense())
        # lx = np.append(lx, )
        # print(self.N_hs * self.T)
        # A = spa.vstack([A, self.ang_cons])

        # lx = np.append(lx, )
        # u = np.append(u, np.tile(self.gmg_ones, self.T))
        # print(ux.shape)
        # l = np.append(l, -1000 * np.ones(self.N_hs * self.T))
        # lx = np.append(lx, np.tile(self.gmg_ones, self.T))
        # print(A.shape, u.shape, l.shape)

        # exit(0)

        # Get index of bounds (all variables)
        bounds_idx = np.arange(A.shape[1])

        # Initialize problem structure
        problem = {}
        problem['P'] = P
        problem['q'] = q
        problem['A'] = A
        problem['l'] = l
        problem['u'] = u
        problem['m'] = A.shape[0]
        problem['n'] = A.shape[1]
        problem['A_nobounds'] = A_nobounds
        problem['l_nobounds'] = l_nobounds
        problem['u_nobounds'] = u_nobounds
        problem['bounds_idx'] = bounds_idx
        problem['lx'] = lx
        problem['ux'] = ux

        return problem

    def _generate_cvxpy_problem(self):
        '''
        Generate QP problem
        '''

        # Dimensions
        nx, nu = self.nx, self.nu
        T = self.T

        # Initial state
        x0 = cvxpy.Parameter(nx)
        x0.value = self.x0

        # variables
        x = cvxpy.Variable((nx, T + 1))
        u = cvxpy.Variable((nu, T))

        # Objective
        cost = cvxpy.quad_form(x[:, T], self.QN)  # Terminal cost
        for i in range(T):
            cost += cvxpy.quad_form(x[:, i], self.Q)     # State cost
            cost += cvxpy.quad_form(u[:, i], self.R)     # Inpout cost
        objective = cvxpy.Minimize(cost)

        # Dynamics
        dynamics = [x[:, 0] == x0]
        for i in range(T):
            dynamics += [x[:, i+1] == self.A @ x[:, i] + self.B @ u[:, i]]

        # State constraints
        state_constraints = []
        for i in range(T + 1):
            state_constraints += [self.xmin <= x[:, i], x[:, i] <= self.xmax]

        # Input constraints
        input_constraints = []
        for i in range(T):
            input_constraints += [self.umin <= u[:, i], u[:, i] <= self.umax]

        problem = cvxpy.Problem(objective,
                                dynamics +
                                state_constraints +
                                input_constraints)

        return problem, (x, u), x0

    def revert_cvxpy_solution(self):
        '''
        Get QP primal and duar variables from cvxpy solution
        '''
        (x_cvx, u_cvx) = self.cvxpy_variables
        constraints = self.cvxpy_problem.constraints
        T = self.T

        # primal solution
        x = np.concatenate((cvxpy.vec(x_cvx).value,
                            cvxpy.vec(u_cvx).value))

        # dual solution
        constraint_values = [constr.dual_value for constr in constraints]
        y = np.array([])

        # Add dynamics
        for i in np.arange(T + 1):
            y = np.append(y, -constraint_values[i])   # Equalities

        # Add state constraints (2 * (T + 1))
        for i in np.arange(T + 1, 3 * (T + 1) - 1, 2):
            y = np.append(y,
                          constraint_values[i + 1] - constraint_values[i])

        # Add input constraints ( 2 * T )
        for i in np.arange(3 * (T + 1), 3 * (T + 1) + 2 * T - 1, 2):
            y = np.append(y,
                          constraint_values[i + 1] - constraint_values[i])

        return x, y

    def _b(self, x):
        """RHS of linear equality constraint in sparse MPC variant"""
        b = np.zeros((self.T + 1) * self.nx)
        b[:self.nx] = -x
        return b

    def update_x0(self, x0_new):
        """
        Update initial state in the problem
        """

        # Update internal x0 parameter
        self.x0 = x0_new

        # Update dynamics in QP problem
        b_new = self._b(self.x0)
        self.qp_problem['l'][:(self.T + 1) * self.nx] = b_new
        self.qp_problem['u'][:(self.T + 1) * self.nx] = b_new
        self.qp_problem['l_nobounds'][:(self.T + 1) * self.nx] = b_new
        self.qp_problem['u_nobounds'][:(self.T + 1) * self.nx] = b_new

        # Update parameter in CVXPY problem
        self.cvxpy_param.value = self.x0


def main():
    mpc = ModelPredictiveControl(n=5, T=5)
    print(mpc.qp_problem)
    P = mpc.qp_problem['P']
    A = mpc.qp_problem['A']
    l = mpc.qp_problem['l']
    u = mpc.qp_problem['u']
    print(P.shape, A.shape, l.shape, u.shape)


if __name__ == '__main__':
    main()
