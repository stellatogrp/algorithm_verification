
def map_linstep_to_ranges(y, u, k, handler):
    iter_to_id_map = handler.iterate_to_id_map
    iter_bound_map = handler.iter_bound_map
    param_bound_map = handler.param_bound_map

    y.get_dim()
    iter_bound_map[y][k]
    uranges = []
    for x in u:
        if x.is_param:
            uranges.append(param_bound_map[x])
        else:
            idx = curr_or_prev(y, x, k, iter_to_id_map)
            uranges.append(iter_bound_map[x][idx])
    return uranges


def curr_or_prev(var1, var2, k, iter_id_map):
    """
    Returning which step of var2 to use
    I.e. if y = LinStep(x), need to know if y^{k} depends on x^k or x^{k-1}
    """
    i1 = iter_id_map[var1]
    i2 = iter_id_map[var2]
    if i1 <= i2:
        return k-1
    return k
