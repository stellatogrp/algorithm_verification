import numpy as np
import cvxpy as cp


def form_one_step_program():
    n = 3
    mu = 2
    L = 20
    gamma = 2/(mu + L)
    R = 1
    I = np.eye(n)

    P = np.diag(mu * np.ones(n))
    P[0][0] = L
    print(P)

    x0 = cp.Variable(n)
    x1 = cp.Variable(n)

    obj = cp.norm(x1, 2)
    constraints = []

    constraints.append(x1 == (I - gamma * P) @ x0)
    constraints.append(cp.norm(x0) <= R)

    problem = cp.Problem(cp.Minimize(obj), constraints)
    result = problem.solve()
    print(result)
    print(x0.value)


def main():
    form_one_step_program()


if __name__ == '__main__':
    main()
