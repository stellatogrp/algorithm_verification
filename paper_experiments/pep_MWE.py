from PEPit import PEP
from PEPit.functions import (
    ConvexFunction,
    # SmoothStronglyConvexFunction,
    SmoothStronglyConvexQuadraticFunction,
)
from PEPit.primitive_steps import proximal_step


def test_quad(mu, L, K, t, r):
    pepit_verbose = 2
    problem = PEP()

    func1 = problem.declare_function(ConvexFunction)
    func2 = problem.declare_function(SmoothStronglyConvexQuadraticFunction, L=L, mu=mu)
    # func2 = problem.declare_function(SmoothStronglyConvexFunction, L=L, mu=mu)
    func = func1 + func2

    xs = func.stationary_point()
    x0 = problem.set_initial_point()

    x = [x0 for _ in range(K + 1)]
    for i in range(K):
        x[i+1], _, _ = proximal_step(x[i] - t * func2.gradient(x[i]), func1, t)

    problem.set_initial_condition((x0 - xs) ** 2 <= r ** 2)

    # Fixed point residual
    problem.set_performance_metric((x[-1] - x[-2]) ** 2)

    # Solve the PEP
    # mosek_params = {
    #     'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-5,
    #     'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-5,
    #     'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-8,
    # }
    # pepit_tau = problem.solve(verbose=pepit_verbose, solver=cp.MOSEK, mosek_params=mosek_params)
    try:
        pepit_tau = problem.solve(verbose=pepit_verbose, wrapper='mosek')
    except AssertionError:
        # print(problem.objective.eval())
        # exit(0)
        pepit_tau = problem.objective.eval()

    print('pepit_tau:', pepit_tau)
    return pepit_tau


def main():
    mu = 20
    L = 100
    K = 10
    t = 0.0125
    r = 12.11
    # r = 1

    test_quad(mu, L, K, t, r)


if __name__ == '__main__':
    main()
