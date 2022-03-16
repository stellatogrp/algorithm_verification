from scipy.sparse import eye

from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.expressions.variable import Variable


def quad_over_lin_canon(expr, args):
    affine_expr = args[0]
    y = args[1]
    # Simplify if y has no parameters.
    if len(y.parameters()) == 0:
        quad_mat = eye(affine_expr.size)/y.value
    else:
        # TODO this codepath produces an intermediate dense matrix.
        # but it should be sparse the whole time.
        quad_mat = eye(affine_expr.size)/y

    if isinstance(affine_expr, Variable):
        return SymbolicQuadForm(affine_expr, quad_mat, expr), []
    else:
        t = Variable(affine_expr.shape)
        return SymbolicQuadForm(t, quad_mat, expr), [affine_expr == t]
