import numpy as np
import cvxpy as cp


def form_and_solve_basic_sdp():
    n = 3
    mu = 2
    L = 20
    R = 1
    gamma = 2 / (mu + L)
    I = np.eye(n)
    P = np.array([[mu, 0, 0], [0, L, 0], [0, 0, (mu + L) / 2]])
    print(P)
    Qb = -(gamma ** 2) * P @ P + 2 * gamma * P - I
    print(Qb)
    ub = np.zeros(n)
    cb = 0

    Qa = I
    ua = np.zeros(n)
    ca = -R ** 2

    form_slemma_sdp(n, Qb, ub, cb, Qa, ua, ca)


def form_offcenter_ball_sdp():
    n = 3
    mu = 2
    L = 20
    R = 1
    gamma = 2 / (mu + L)
    I = np.eye(n)
    P = np.array([[mu, 0, 0], [0, L, 0], [0, 0, (mu + L) / 2]])
    # print(P)
    Qb = -(gamma ** 2) * P @ P + 2 * gamma * P - I
    ub = np.zeros(n)
    cb = 0

    # generate unit ball off center
    epsilon = .1
    random_point = np.random.normal(0, 1, 3)
    x_center = random_point / np.linalg.norm(random_point)
    print('center:', x_center, 'norm:', np.linalg.norm(x_center))

    Qa = I
    ua = -2 * x_center
    ca = -epsilon ** 2 + np.inner(x_center, x_center)

    form_slemma_sdp(n, Qb, ub, cb, Qa, ua, ca)


def form_quad_obj_sdp():
    n = 3
    mu = 1
    L = 10
    R = 1
    gamma = 2 / (mu + L)
    I = np.eye(n)
    P = np.array([[mu, 0, 0], [0, L, 0], [0, 0, (mu + L) / 2]])

    Qb = .5 * (-(gamma ** 2) * P @ P @ P + 2 * gamma * P @ P - P)
    print(Qb)
    ub = np.zeros(n)
    cb = 0

    Qa = I
    ua = np.zeros(n)
    ca = -R ** 2

    form_slemma_sdp(n, Qb, ub, cb, Qa, ua, ca)


def form_slemma_sdp(n, Qb, ub, cb, Qa, ua, ca):
    eta = cp.Variable()
    lambd = cp.Variable()

    M = cp.Variable((n + 1, n + 1), symmetric=True)
    constraints = [M >> 0, lambd >= 0]

    constraints.append(M[0:n, 0:n] == Qb + lambd * Qa)
    constraints.append(M[0:n, n] == .5 * (ub + lambd * ua))
    constraints.append(M[n][n] == cb - eta + lambd * ca)

    problem = cp.Problem(cp.Maximize(eta), constraints)
    result = problem.solve()

    print('sdp result:', result)
    print(constraints[0].dual_value)


def main():
    # form_and_solve_basic_sdp()
    # form_offcenter_ball_sdp()
    form_quad_obj_sdp()


if __name__ == '__main__':
    main()
