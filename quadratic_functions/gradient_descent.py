
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
