from scipy.sparse import eye, bmat

from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.hstack import hstack
#  from cvxpy.utilities.shape import mul_shapes_promote
#  from cvxpy.atoms.affine.reshape import reshape


def mulexpression_canon(expr, args):
    lhs = expr.args[0]
    rhs = expr.args[1]

    # If it is not a product of variables, do not do anything.
    # TODO (bart): deal with parametric problem
    if lhs.is_constant() or rhs.is_constant():
        return expr, []

    if lhs.ndim > 1 or rhs.ndim > 1:
        raise ValueError("Product works only between vectors")

    # TODO (bart): fix this in multiple dimensions
    # Also, this is not the right place to put errors
    # They should go in the reduction.accepts(problem)
    #  lhs_shape, rhs_shape, _ = mul_shapes_promote(lhs.shape, rhs.shape)
    #  lhs = reshape(lhs, lhs_shape)
    #  rhs = reshape(rhs, rhs_shape)

    constraints = []

    if isinstance(lhs, Variable):
        lh_var = lhs
    else:
        lh_var = Variable(lhs.shape)
        constraints += [lhs == lh_var]
    if isinstance(rhs, Variable):
        rh_var = rhs
    else:
        rh_var = Variable(rhs.shape)
        constraints += [rhs == rh_var]

    # Write constraints product x @ y as
    # z = (x, y)
    # and 0.5 * z' * M * z where
    # M = [0, I
    #      I, 0]
    # and I is the identity matrix with appropriate dimensions
    z = Variable(lhs.shape[0] + rhs.shape[0])
    constraints += [z == hstack([lh_var, rh_var])]
    quad_mat = 0.5 * bmat([[None, eye(lhs.shape[0])],
                           [eye(rhs.shape[0]), None]])

    return SymbolicQuadForm(z, quad_mat, expr), constraints
