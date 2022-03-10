import numpy as np
import cvxpy as cp
from qcqp2quad_form.quad_extractor import QuadExtractor

# Random problem data
n = 3
m = 2
A = np.random.randn(m, n)
b = np.random.rand(m)
x = cp.Variable(n)
y = cp.Variable(m)
P = np.random.randn(n, n)
P = P @ P.T - 5 * np.eye(n)
q = np.random.rand(n)
r = np.random.randn()


problem = cp.Problem(cp.Minimize(cp.quad_form(x, P) + q @ x + r),
                     [A @ x <= b, cp.quad_form(x, A.T @ A) <= r,
                      cp.sum_squares(A @ x - b) == r,
                      x @ (A.T @ y) <= r])
quad_extractor = QuadExtractor(problem)
data_objective = quad_extractor.extract_objective()
data_constraints = quad_extractor.extract_constraints()
