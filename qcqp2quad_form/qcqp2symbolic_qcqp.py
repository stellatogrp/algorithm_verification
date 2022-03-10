from cvxpy.constraints import Equality, Inequality, NonNeg, NonPos, Zero
from cvxpy.reductions.canonicalization import Canonicalization
from cvxpy.reductions.cvx_attr2constr import convex_attributes

#  from cvxpy.reductions.qp2quad_form.atom_canonicalizers import (
#      CANON_METHODS as qp_canon_methods,)
from qcqp2quad_form.atom_canonicalizers import (CANON_METHODS as qcqp_canon_methods,)


def accepts(problem):
    """
    Problems with quadratic objectives and constraints (equality and inequality),
    are accepted by this reduction
    """
    return (problem.objective.expr.is_qpwa()
            and not set(['PSD', 'NSD']).intersection(convex_attributes(
                                                     problem.variables()))
            and all((type(c) in (Inequality, Equality,
                                 Zero, NonPos, NonNeg) and
                    c.expr.is_quadratic()) for c in problem.constraints))


class Qcqp2SymbolicQcqp(Canonicalization):
    """
    Reduces a quadratic problem to a problem that consists of affine
    expressions and symbolic quadratic forms.
    """

    def __init__(self, problem=None) -> None:
        super(Qcqp2SymbolicQcqp, self).__init__(
            problem=problem, canon_methods=qcqp_canon_methods)

    def accepts(self, problem):
        """
        Problems with quadratic, piecewise affine objectives,
        piecewise-linear constraints inequality constraints, and
        affine equality constraints are accepted.
        """
        return accepts(problem)

    def apply(self, problem):
        """Converts a QP to an even more symbolic form."""
        if not self.accepts(problem):
            raise ValueError("Cannot reduce problem to symbolic QP")
        return super(Qcqp2SymbolicQcqp, self).apply(problem)
