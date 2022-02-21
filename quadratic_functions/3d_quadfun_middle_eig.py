import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm, trange
from generate_quadratic_function import generate_quadfun_grad
from gradient_descent import *


def generate_uniform_sphere(n, num_points, R=1):
    normal_vals = np.random.normal(size=(num_points, n))
    scaled_vals = normal_vals / np.linalg.norm(normal_vals, axis=1, keepdims=True)
    # print(scaled_vals, np.linalg.norm(scaled_vals, axis=1))
    return scaled_vals * R


def R3_quadfun_driver():
    n = 3
    mu = 2
    L = 20
    R = 1
    # gamma = 2 / (mu + L)
    gamma = 0.05
    num_points = 200
    num_iter = 1

    initial_iterates = generate_uniform_sphere(n, num_points, R=R)
    # print(initial_iterates, np.linalg.norm(initial_iterates, axis=1))

    P = np.zeros(shape=(n, n))
    P[0][0] = L
    P[n-1][n-1] = mu
    num_test_Pvals = 100

    middle_eigvals = np.arange(mu, L, (L - mu) / num_test_Pvals)
    middle_eigvals = np.append(middle_eigvals, L)
    print(middle_eigvals)
    q = np.zeros(n)

    min_performances = []
    avg_performances = []
    max_performances = []

    for e in tqdm(middle_eigvals):
        P[1][1] = e
        f, grad = generate_quadfun_grad(P, q)
        final_norms = []
        for i in range(num_points):
            x_0 = initial_iterates[i]
            iterates = grad_descent(grad, x_0, gamma, num_iter)
            x_N = iterates[-1]
            final_norms.append(np.sqrt(np.inner(x_N, x_N)))
        min_performances.append(np.min(final_norms))
        avg_performances.append(np.mean(final_norms))
        max_performances.append(np.max(final_norms))

    fig, ax = plt.subplots(1)

    ax.scatter(middle_eigvals, min_performances, label='min')
    ax.scatter(middle_eigvals, avg_performances, label='avg')
    ax.scatter(middle_eigvals, max_performances, label='max')
    # print(max_performances)
    title = 'Minimizing quadratic in R^3 with diff x^0 values,\n mu={}, L={}, R={}, gamma={:.3f}'.format(mu, L, R, gamma)
    ax.set_title(title)
    ax.set_xlabel('middle eigenvalue')
    ax.set_ylabel('average |x^1 - x^\star|_2')

    plt.legend()
    plt.show()


def main():
    R3_quadfun_driver()


if __name__ == '__main__':
    main()
