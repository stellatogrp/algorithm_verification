from algocert.variables.iterate import Iterate
from algocert.basic_algorithm_steps.max_with_vec_step import MaxWithVecStep
from algocert.basic_algorithm_steps.min_with_vec_step import MinWithVecStep


def box_proj_step_canon(step):
    '''
        Convert a box proj step -> max with l followed by min with u
    '''
    # D = step.get_lhs_matrix()
    # A = step.get_rhs_matrix()
    # b = step.get_rhs_const_vec()
    # y = step.get_output_var()
    # u = step.get_input_var()
    # u_dim = 0
    # for x in u:
    #     u_dim += x.get_dim()
    # name = y.get_name()
    # new_name = name + '_block'
    # u_block = Iterate(u_dim, name=new_name)
    # step1 = BlockStep(u_block, u)
    # step2 = LinearStep(y, u_block, A=A, D=D, b=b)
    # return [u_block], [step1, step2]
    y = step.get_output_var()
    x = step.get_input_var()
    l = step.get_lower_bound_vec()
    u = step.get_upper_bound_vec()
    name = y.get_name()
    new_name = name + '_tilde'
    y_tilde = Iterate(y.get_dim(), name=new_name)
    step1 = MaxWithVecStep(y_tilde, x, l=l)
    step2 = MinWithVecStep(y, y_tilde, u=u)
    return [y_tilde], [step1, step2]
