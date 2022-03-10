from cvxpy.atoms.quad_form import SymbolicQuadForm
from cvxpy.expressions.variable import Variable


def quad_form_canon(expr, args):
    affine_expr = expr.args[0]
    P = expr.args[1]
    if isinstance(affine_expr, Variable):
        return SymbolicQuadForm(affine_expr, P, expr), []
    else:
        t = Variable(affine_expr.shape)
        return SymbolicQuadForm(t, P, expr), [affine_expr == t]

