from algocert.basic_algorithm_steps.basic_linear_step import BasicLinearStep
from algocert.basic_algorithm_steps.block_step import BlockStep
from algocert.variables.iterate import Iterate


def linear_step_canon(step):
    '''
        Convert a higher level linear step -> a block step followed by a homogenized linear step
    '''
    D = step.get_lhs_matrix()
    Dinv = step.get_lhs_matrix_inv()
    A = step.get_rhs_matrix()
    b = step.get_rhs_const_vec()
    y = step.get_output_var()
    u = step.get_input_var()
    u_dim = 0
    for x in u:
        u_dim += x.get_dim()
    name = y.get_name()
    new_name = name + '_block'
    u_block = Iterate(u_dim, name=new_name)
    step1 = BlockStep(u_block, u)
    step2 = BasicLinearStep(y, u_block, A=A, D=D, b=b, Dinv=Dinv)
    return [u_block], [step1, step2]