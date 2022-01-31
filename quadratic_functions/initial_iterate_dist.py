import numpy as np
import matplotlib.pyplot as plt

from generate_quadratic_function import *


def grad_descent_step(grad, x_0, gamma=0.01):
    grad_step = grad(x_0)
    return x_0 - gamma * grad_step


def grad_descent(grad, x_0, gamma, num_iter, tol=1e-8):
    iterates = []
    for i in range(num_iter):
        iterates.append(x_0)
        x_0 = grad_descent_step(grad, x_0, gamma)
    iterates.append(x_0)
    return iterates


def generate_uniform_circle(num_points, R):
    '''
        Generates num_points number of points on the circle in the plane with radius R
    '''
    theta_vals = np.random.uniform(0, 2 * np.pi, size=num_points)
    x_vals = np.sqrt(R) * np.cos(theta_vals)
    y_vals = np.sqrt(R) * np.sin(theta_vals)
    return x_vals, y_vals


def generate_uniform_sphere(n, num_points, R):
    normal_vals = np.random.normal(size=(num_points, n))
    scaled_vals = normal_vals / np.linalg.norm(normal_vals, axis=1, keepdims=True)
    # print(scaled_vals, np.linalg.norm(scaled_vals, axis=1))
    return scaled_vals


def mult_x0_plot_driver():
    n = 2
    mu = 2
    L = 20
    R = 1
    num_points = 200
    num_iter = 1
    gamma = 2 / (mu + L)

    # f, grad = generate_simple_quadfun(n, mu, L)
    f, grad = generate_skewed_simple_quadfun(n, mu, L)

    test_x_vals, test_y_vals = generate_uniform_circle(num_points, R)

    final_norms = []
    for i in range(num_points):
        x_0 = np.array([test_x_vals[i], test_y_vals[i]])
        iterates = grad_descent(grad, x_0, gamma, num_iter)
        print(iterates)
        x_N = iterates[-1]
        print(f(x_N))
        final_norms.append(np.sqrt(np.inner(x_N, x_N)))

    print(final_norms)

    ax = plt.axes(projection='3d')
    ax.scatter3D(test_x_vals, test_y_vals, np.zeros(num_points))
    ax.scatter3D(test_x_vals, test_y_vals, final_norms)

    ax.set_zlim(0, 1)
    plt.show()


def high_dim_generation_driver():
    n = 4
    mu = 2
    L = 20
    R = 1
    num_points = 200
    num_iter = 1
    gamma = 2 / (mu + L)

    initial_points = generate_uniform_sphere(n, num_points, R)
    print(initial_points[num_points-1])

    f, grad = generate_simple_quadfun(n, mu, L)
    # f, grad = generate_rand_spectrum_quadfun(n, mu, L)
    final_norms = []
    for i in range(num_points):
        x_0 = initial_points[i]
        iterates = grad_descent(grad, x_0, gamma, num_iter)
        x_N = iterates[-1]
        # print(x_N)
        # print(np.sqrt(np.inner(x_N, x_N)))
        # print(f(x_N))
        final_norms.append(np.sqrt(np.inner(x_N, x_N)))
    print(final_norms)


def main():
    # mult_x0_plot_driver()
    high_dim_generation_driver()


if __name__ == '__main__':
    main()
