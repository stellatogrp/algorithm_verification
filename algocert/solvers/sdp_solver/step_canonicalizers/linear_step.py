import numpy as np


def linear_step_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars, add_RLT, kwargs):
    '''
        Convert a higher level linear step -> a block step followed by a homogenized linear step
    '''
    print('here')

    exit(0)
    # D = step.get_lhs_matrix()
    # Dinv = step.get_lhs_matrix_inv()
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
    # step2 = BasicLinearStep(y, u_block, A=A, D=D, b=b, Dinv=Dinv)
    # return [u_block], [step1, step2]


def linear_step_bound_canon(steps, i, curr, prev, iter_id_map, param_vars, param_outerproduct_vars):
    step = steps[i]
    #  prev_step = steps[i - 1]
    u = step.get_input_var()  # remember this is a list of vars
    y = step.get_output_var()
    A = step.get_rhs_matrix()
    #  D = step.get_lhs_matrix()
    Dinv = step.get_lhs_matrix_inv()
    b = step.get_rhs_const_vec()

    DinvA = Dinv @ A
    Dinvb = (Dinv @ b).reshape((-1, 1))

    u_lower = []
    u_upper = []
    # print(u)
    for x in u:
        if x.is_param:
            u_lower.append(param_vars[x].get_lower_bound())
            u_upper.append(param_vars[x].get_upper_bound())
        else:
            if curr_or_prev(y, x, iter_id_map) == 0:
                # handlers_to_use.append(prev)
                x_bound_obj = prev.iterate_vars[x]
                u_lower.append(x_bound_obj.get_lower_bound())
                u_upper.append(x_bound_obj.get_upper_bound())
            else:
                # handlers_to_use.append(curr)
                x_bound_obj = curr.iterate_vars[x]
                u_lower.append(x_bound_obj.get_lower_bound())
                u_upper.append(x_bound_obj.get_upper_bound())

    u_lower_bound = np.vstack(u_lower)
    u_upper_bound = np.vstack(u_upper)
    # print(u_lower_bound, u_upper_bound)

    lower_y, upper_y = lin_bound_map(u_lower_bound, u_upper_bound, DinvA)

    # print(lower_y + Dinvb)
    # print(upper_y + Dinvb)

    curr.iterate_vars[y].set_lower_bound(lower_y + Dinvb)
    curr.iterate_vars[y].set_upper_bound(upper_y + Dinvb)


def lin_bound_map(l, u, A):
    A = A.toarray()
    (m, n) = A.shape
    l_out = np.zeros(m)
    u_out = np.zeros(m)
    for i in range(m):
        lower = 0
        upper = 0
        for j in range(n):
            if A[i][j] >= 0:
                lower += A[i][j] * l[j]
                upper += A[i][j] * u[j]
            else:
                lower += A[i][j] * u[j]
                upper += A[i][j] * l[j]
        l_out[i] = lower
        u_out[i] = upper

    return np.reshape(l_out, (m, 1)), np.reshape(u_out, (m, 1))


def curr_or_prev(var1, var2, iter_id_map):
    """
    Returning 0 means to use the previous iteration handler for var2
    Returning 1 means to use the current
    """
    i1 = iter_id_map[var1]
    i2 = iter_id_map[var2]
    if i1 <= i2:
        return 0
    return 1
