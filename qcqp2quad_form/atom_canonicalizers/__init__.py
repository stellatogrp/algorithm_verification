from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.quad_form import QuadForm
from cvxpy.atoms.quad_over_lin import quad_over_lin

#  from cvxpy.atoms.affine.index import special_index
#  from cvxpy.reductions.dcp2cone.atom_canonicalizers import (
#      CANON_METHODS as CONE_METHODS,)
#  from cvxpy.reductions.qp2quad_form.atom_canonicalizers.huber_canon import *
#  from cvxpy.reductions.qp2quad_form.atom_canonicalizers.power_canon import *
#  from cvxpy.reductions.qp2quad_form.atom_canonicalizers.quad_form_canon import *
#  from cvxpy.reductions.qp2quad_form.atom_canonicalizers.quad_over_lin_canon import *
#  from cvxpy.transforms.indicator import indicator
from qcqp2quad_form.atom_canonicalizers.quad_form_canon import quad_form_canon
from qcqp2quad_form.atom_canonicalizers.quad_over_lin_canon import quad_over_lin_canon
from qcqp2quad_form.atom_canonicalizers.mulexpression_canon import mulexpression_canon
CANON_METHODS = {}

# TODO: remove pwl canonicalize methods, use EliminatePwl reduction instead

# reuse cone canonicalization methods
#  CANON_METHODS[abs] = CONE_METHODS[abs]
#  CANON_METHODS[cumsum] = CONE_METHODS[cumsum]
#  CANON_METHODS[maximum] = CONE_METHODS[maximum]
#  CANON_METHODS[minimum] = CONE_METHODS[minimum]
#  CANON_METHODS[sum_largest] = CONE_METHODS[sum_largest]
#  CANON_METHODS[max] = CONE_METHODS[max]
#  CANON_METHODS[min] = CONE_METHODS[min]
#  CANON_METHODS[norm1] = CONE_METHODS[norm1]
#  CANON_METHODS[norm_inf] = CONE_METHODS[norm_inf]
#  CANON_METHODS[indicator] = CONE_METHODS[indicator]
#  CANON_METHODS[special_index] = CONE_METHODS[special_index]

# canonicalizations that are different for QPs
CANON_METHODS[quad_over_lin] = quad_over_lin_canon
#  CANON_METHODS[power] = power_canon
#  CANON_METHODS[huber] = huber_canon
CANON_METHODS[QuadForm] = quad_form_canon
CANON_METHODS[MulExpression] = mulexpression_canon
