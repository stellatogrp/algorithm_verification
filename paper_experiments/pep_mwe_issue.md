Hello @AdrienTaylor and @NizarBousselmi,

First, thank you for creating this library!
I am working with @bstellato on verification methods specific to parametric QPs [here](https://github.com/stellatogrp/algorithm_verification/tree/main) and used PEPit for comparison purposes in our [pape](https://arxiv.org/abs/2403.03331).
While working with the `StrongSmoothlyConvexQuadraticFunction` object, I ran into numerical issues with the following example:

```
import cvxpy as cp
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

```

### Solving with the mosek wrapper directly.

In this case, with
```
pepit_tau = problem.solve(verbose=pepit_verbose, wrapper='mosek')
```
I notice that the code errors out with an assertion error in the feasibility check for the `PEP` object:
```
 File "/opt/homebrew/Caskroom/miniconda/base/envs/algover_test/lib/python3.9/site-packages/PEPit/pep.py", line 680, in check_feasibility
    assert wc_value == self.objective.eval()
AssertionError
```
It seems that the final value from Mosek is extracted incorrectly when performing the check as the `wc_value` is 0 but `self.objective.eval()` correctly retrieves the answer from Mosek.

### Solving through CVXPY
However, when solving through CVXPY to try other solvers, I noticed numerical issues.
When running the same instance with an uncommented line, either
```
pepit_tau = problem.solve(verbose=pepit_verbose, solver=cp.MOSEK)
```
or
```
pepit_tau = problem.solve(verbose=pepit_verbose, solver=cp.CLARABEL)
```
the solvers error out with numerical issues when using the default tolerances.

For example, Mosek reports
```
(CVXPY) Mar 25 06:11:46 PM: Interior-point solution summary
(CVXPY) Mar 25 06:11:46 PM:   Problem status  : UNKNOWN
(CVXPY) Mar 25 06:11:46 PM:   Solution status : UNKNOWN
(CVXPY) Mar 25 06:11:46 PM:   Primal.  obj: -2.5809495701e-01   nrm: 2e+02    Viol.  con: 5e-04    var: 3e-06    barvar: 0e+00
(CVXPY) Mar 25 06:11:46 PM:   Dual.    obj: -2.6290013277e-01   nrm: 3e+03    Viol.  con: 0e+00    var: 1e-03    barvar: 1e-05
```
and Clarabel reports
```
iter    pcost        dcost       gap       pres      dres      k/t        μ       step
---------------------------------------------------------------------------------------------
  0  -2.1533e-04  -1.5070e+02  1.51e+02  3.75e-01  3.12e-02  1.00e+00  3.20e+01   ------
  1  -2.1533e-04  -1.5070e+02  1.51e+02  3.75e-01  3.12e-02  1.00e+00  3.20e+01  0.00e+00
---------------------------------------------------------------------------------------------
Terminated with status = NumericalError
```

It seems there is some numeric issue in the problem formulation that gets fed into cvxpy. When using a more general class such as `SmoothStronglyConvexFunction` for example, all of these issues disappear and the three methods agree.
