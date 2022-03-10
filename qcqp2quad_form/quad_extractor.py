from cvxpy.utilities.coeff_extractor import CoeffExtractor
from cvxpy.reductions import InverseData
#  from cvxpy.reductions.qp2quad_form.qp2symbolic_qp import Qp2SymbolicQp
from qcqp2quad_form.qcqp2symbolic_qcqp import Qcqp2SymbolicQcqp
from cvxpy.cvxcore.python import canonInterface


class QuadExtractor():
    def __init__(self, problem):
        reduction = Qcqp2SymbolicQcqp()
        self.symb_problem, self.symb_inverse = reduction.apply(problem)
        import ipdb; ipdb.set_trace()
        self.inverse_data = InverseData(self.symb_problem)
        self.extractor = CoeffExtractor(self.inverse_data)

    def extract_expression(self, expr):
        if expr.is_affine():
            q_t = self.extractor.affine(expr)
            P = None
        elif expr.is_quadratic():
            P_t, q_t = self.extractor.quad_form(expr)
            P, _ = canonInterface.get_matrix_from_tensor(
                P_t, None, self.extractor.x_length, with_offset=False)

        q, r = canonInterface.get_matrix_from_tensor(
            q_t, None, self.extractor.x_length, with_offset=True)
        q = q.toarray()
        if r.size == 1:
            r = r.item()
        return {'P': P, 'q': q, 'r': r}

    def extract_objective(self):
        data = self.extract_expression(self.symb_problem.objective.expr)
        data['q'] = data['q'].flatten()
        return data

    def extract_constraints(self):
        constraint_coeffs = []
        for c in self.symb_problem.constraints:
            constraint_coeffs += [self.extract_expression(c.expr)]
        return constraint_coeffs
