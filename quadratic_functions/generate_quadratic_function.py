import numpy as np
from scipy.stats import ortho_group


def generate_quadfun_grad(P, q):
    '''
        Generates quadratic function of form (1/2) x.T @ P @ x + q.T @ x and gradient
    '''
    def f(x):
        return (1 / 2) * x.T @ P @ x + q.T @ x

    def grad(x):
        return P @ x + q

    return f, grad


def generate_rand_spectrum_quadfun(n, mu, L):
    '''
        Generates quadratic function of dimension n such that:
            - The Hessian has min/max eigenvalues equal to mu/L, and all others random in the range
            - Linear term q = 0
    '''
    if n <= 2:
        return generate_simple_quadfun(n, mu, L)

    spectrum = np.random.uniform(mu, L, size=n-2)
    sorted_spectrum = np.sort(spectrum)[::-1]
    P = np.diag(mu * np.ones(n))
    P[0][0] = L
    P[1:n-1, 1:n-1] = np.diag(sorted_spectrum)
    print(P)
    q = np.zeros(n)
    return generate_quadfun_grad(P, q)


def generate_simple_quadfun(n, mu, L):
    '''
        Generates quadratic function of dimension n such that:
            - The Hessian has all eigenvalues equal to mu except for the largest, which is equal to L
            - Linear term q = 0
    '''
    P = np.diag(mu * np.ones(n))
    P[0][0] = L
    # P[1][1] = L
    print(P)
    q = np.zeros(n)
    return generate_quadfun_grad(P, q)


def generate_skewed_simple_quadfun(n, mu, L):
    P = np.diag(mu * np.ones(n))
    P[0][0] = L
    U = ortho_group.rvs(n)

    final_P = U @ P @ U.T
    print(final_P)

    q = np.zeros(n)

    return generate_quadfun_grad(final_P, q)

