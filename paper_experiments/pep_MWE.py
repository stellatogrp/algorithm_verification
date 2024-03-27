from PEPit import PEP
from PEPit.functions import (
    ConvexFunction,
    SmoothStronglyConvexQuadraticFunction,
)
from PEPit.primitive_steps import proximal_step


def test_quad(mu, L, K, t, r):
    pepit_verbose = 2
    problem = PEP()

    # proximal gradient descent for sum of quadratic and convex function
    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(SmoothStronglyConvexQuadraticFunction, L=L, mu=mu)
    func = func1 + func2

    xs = func.stationary_point()
    x0 = problem.set_initial_point()

    x = [x0 for _ in range(K + 1)]
    for i in range(K):
        x[i+1], _, _ = proximal_step(x[i] - t * func2.gradient(x[i]), func1, t)

    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)

    # Fixed point residual
    problem.set_performance_metric((x[-1] - x[-2]) ** 2)

    pepit_tau = problem.solve(verbose=pepit_verbose, wrapper='mosek')
    # pepit_tau = problem.solve(verbose=pepit_verbose, solver=cp.MOSEK)
    # pepit_tau = problem.solve(verbose=pepit_verbose, solver=cp.CLARABEL)

    print('pepit_tau:', pepit_tau)
    return pepit_tau

mu = 1
L = 10
K = 6
t = 0.0125
r = 10

test_quad(mu, L, K, t, r)
