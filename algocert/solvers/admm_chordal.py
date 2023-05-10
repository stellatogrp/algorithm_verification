import cvxpy as cp
# import jax.scipy as jsp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as spa
from tqdm import trange
import jax.numpy as jnp
import jax.lax as lax
from functools import partial
from algocert.solvers.sdp_cgal_solver.lanczos import lanczos
from jax.experimental import sparse
import time
import jax.scipy as jsp
from jax import jit, vmap


def chordal_solve(A_mat_list, C, l, u, chordal_list_of_lists, sigma=1, rho=1, k=500):
    """
    solves the following sdp with chordal sparsity

    min tr(C X)
        s.t.  l_i <= tr(A_i X) <= u_i i=1, ..., m
              X_j = E_j X E_j.T
              X_j >> 0 j=1,...,p

    The E_j matrices can be formed from chordal_list_of_lists

    Each E_j has shape (r, n) where X has shape (n, n)
    - each row of each E_j has exactly one 1 and the rest zeros
    - the one corresponds to that variable being in the chord
    """
    num_chords = len(chordal_list_of_lists)

    # convert to separable form
    c = vec_symm(C)
    A = convert_A_mat_list_2_A(A_mat_list)

    psd_sizes = []
    for i in range(num_chords):
        psd_sizes.append(chordal_list_of_lists[i].size)

    # create the B matrix
    n_orig = A_mat_list[0].shape[0]
    B = create_B(A, chordal_list_of_lists, n_orig)
    
    # call the separable solver
    algo_out = solve_separable_sdp(c, B, l, u, psd_sizes, sigma, rho, k)
    z_final, iter_losses, z_all = algo_out
    return z_final, iter_losses, z_all


def create_B(A, chordal_list_of_lists, n_orig):
    m, n = A.shape
    mat_list = [A]
    p = len(chordal_list_of_lists)
    for i in range(p):
        curr_mat = chords_2_mat(chordal_list_of_lists[i], n_orig)
        mat_list.append(curr_mat)
    B = jnp.vstack(mat_list)
    return B


def chords_2_mat(chordal_vec, n):
    r = chordal_vec.size
    E = jnp.zeros((r, n))
    E = E.at[jnp.arange(r), chordal_vec].set(1)

    H = vec_symm_kron(E)
    
    return H


def vec_symm_kron(E):
    """
    given a matrix E, this creates a matrix H s.t.
        Y = E X E.T
        vec_symm(Y) = H_symm vec_symm(X)

    E has shape (r, n)
    H_symm will have shape (r * (r + 1) / 2, n * (n + 1) / 2)
    """
    r, n = E.shape
    H = jnp.kron(E, E)

    # get column indices that correspond to diagonals, upper triangular
    a = np.arange(n ** 2).reshape(n, n)
    diag_n_cols = a[jnp.diag_indices(n)]
    triu_n_cols = a[jnp.triu_indices(n)]

    # divide non-diagonal columns by sqrt(2)
    H_symm = H / jnp.sqrt(2)
    H_diag_col_vals = H_symm[:, diag_n_cols]
    H_symm = H_symm.at[:, diag_n_cols].set(H_diag_col_vals * jnp.sqrt(2))

    # get row indices that correspond to diagonals, upper triangular
    b = np.arange(r ** 2).reshape(r, r)
    diag_r_rows = b[jnp.diag_indices(r)]
    triu_r_rows = b[jnp.triu_indices(r)]

    # multiply non-diagonal rows by sqrt(2)
    H_symm = H_symm * jnp.sqrt(2)
    H_diag_row_vals = H_symm[diag_r_rows, :]
    H_symm = H_symm.at[diag_r_rows, :].set(H_diag_row_vals / jnp.sqrt(2))

    # cut the lower triangular rows and cols
    H_symm = H_symm[triu_r_rows[:, jnp.newaxis], triu_n_cols]
    return H_symm


def convert_A_mat_list_2_A(A_mat_list):
    m = len(A_mat_list)
    n = A_mat_list[0].shape[0]
    nc2 = int(n * (n + 1) / 2)
    A = jnp.zeros((m, nc2))
    for i in range(m):
        A = A.at[i, :].set(vec_symm(A_mat_list[i]))
    return A


def solve_separable_sdp(c, B, l, u, psd_sizes, sigma, rho, k):
    """
    solves the following separable sdp

    min c.T x
        s.t.  Bx = w
              w in C

    C = box(l, u) x psd_cone(psd_sizes[0]) x ... x psd_cone(psd_sizes[p - 1])
    """
    m, n = B.shape

    # create the projection
    proj = create_algocert_projection(l, u, psd_sizes)

    # create the linear system factorization
    # M = sigma * jnp.eye(n) + B.T @ jnp.diag(rho) @ B
    M = sigma * jnp.eye(n) + B.T @ B
    factor = jsp.linalg.lu_factor(M)

    # z0 = jnp.array(np.zeros(n + 2 * m))
    z0 = jnp.array(np.zeros(n + m))
    algo_out = k_steps_eval_osqp(k, z0, c, factor, proj, B, rho, sigma, jit=True)
    return algo_out


def create_algocert_projection(l, u, psd_sizes):
    """
    creates a function that does the following projection
    let p = len(psd_sizes)
    
    consider w = proj(u, v_1,...,v_p)
    then
        w = (z, x_1, ..., x_p)
        z = min(max(u, l), u)
        x_j = proj_psd(v_j) j=1,...,p
    """
    cones = dict(s=psd_sizes)
    psd_projections = create_projection_fn(cones, 0)
    def full_proj(z):
        z_box, z_psd = z[:l.size], z[l.size:]
        z_box_proj = jnp.clip(z_box, a_min=l, a_max=u)
        z_psd_proj = psd_projections(z_psd)
        z_full_proj = jnp.concatenate([z_box_proj, z_psd_proj])
        return z_full_proj
    return full_proj


def fixed_point_osqp(z, factor, B, c, proj, rho, sigma):
    # z = (x, y, w) w is the z variable in osqp terminology
    m, n = B.shape
    x, y, w = z[:n], z[n:n + m], z[n + m:]
    # c, l, u = q[:n], q[n:n + m], q[n + m:]

    # update (x, nu)
    rhs = sigma * x - c + B.T @ (rho * w - y)
    x_next = lin_sys_solve(factor, rhs)
    nu = rho * (B @ x_next - w) + y

    # update w_tilde
    w_tilde = w + (nu - y) / rho

    # update w
    # w_next = jnp.clip(w_tilde + y / rho, a_min=l, a_max=u)
    w_next = proj(w_tilde + y / rho)

    # update y
    y_next = y + rho * (w_tilde - w_next)

    # concatenate into the fixed point vector
    z_next = jnp.concatenate([x_next, y_next, w_next])

    return z_next


# def fixed_point_osqp(fp_vec, factor, proj, B, c, rho, sigma):
#     # z = (x, y, w) w is the z variable in osqp terminology
#     m, n = B.shape
#     x, y = fp_vec[:n], fp_vec[n:n + m]
#     w = fp_vec[n + m:]

#     # c, l, u = q[:n], q[n:n + m], q[n + m:]

#     # update (x, nu)
#     rhs = sigma * x - c + B.T @ (rho * w - y)
#     x_next = lin_sys_solve(factor, rhs)
#     nu = rho * (B @ x_next - w) + y

#     # update z_tilde
#     z_tilde = z + (nu - y) / rho

#     # update x with psd projection


#     # update z with box projection
#     z_next = jnp.clip(z_tilde + y / rho, a_min=l, a_max=u)

#     # update w
#     w_next = w + sigma * (x_tilde - x_next)

#     # update y
#     y_next = y + rho * (z_tilde - z_next)

#     # concatenate into the fixed point vector
#     fp_vec_next = jnp.concatenate([x_next, z_next, y_next, w_next])

#     return fp_vec_next

def fp_eval_osqp(i, val, factor, proj, B, c, rho, sigma):
    z, loss_vec, z_all = val
    # z_next = fixed_point_osqp(z, factor, A, q, rho, sigma)
    z_next = fixed_point_osqp(z, factor, B, c, proj, rho, sigma)
    diff = jnp.linalg.norm(z_next - z)
    loss_vec = loss_vec.at[i].set(diff)
    z_all = z_all.at[i, :].set(z_next)
    return z_next, loss_vec, z_all

def k_steps_eval_osqp(k, z0, c, factor, proj, B, rho, sigma, jit):
    iter_losses = jnp.zeros(k)
    m, n = B.shape

    # initialize z_init
    z_init = jnp.zeros((n + 2 * m))
    z_init = z_init.at[:m + n].set(z0)
    w = B @ z0[:n]
    z_init = z_init.at[m + n:].set(w)

    z_all_plus_1 = jnp.zeros((k + 1, z_init.size))
    z_all_plus_1 = z_all_plus_1.at[0, :].set(z_init)
    fp_eval_partial = partial(fp_eval_osqp,
                              factor=factor,
                              proj=proj,
                              B=B,
                              c=c,
                              rho=rho,
                              sigma=sigma
                              )
    z_all = jnp.zeros((k, z_init.size))
    val = z_init, iter_losses, z_all
    start_iter = 0
    if jit:
        out = lax.fori_loop(start_iter, k, fp_eval_partial, val)
    else:
        out = python_fori_loop(start_iter, k, fp_eval_partial, val)
    z_final, iter_losses, z_all = out
    z_all_plus_1 = z_all_plus_1.at[1:, :].set(z_all)
    return z_final, iter_losses, z_all_plus_1

def lin_sys_solve(factor, b):
    """
    solves the linear system
    Ax = b
    where factor is the lu factorization of A
    """
    return jsp.linalg.lu_solve(factor, b)

def cgal(A_op, C_op, A_star_op, b, alpha, norm_A, rescale_obj, rescale_feas, cgal_iters, m, n, beta0=1, y_max=jnp.inf,
         lobpcg_iters=1000, lobpcg_tol=1e-30, warm_start_v=True, jit=True, lightweight=False):
    """
    jax implementation of the cgal algorithm to solve
    min Tr(CX)
        s.t. Tr(A_i X) = b_i i=1, ..., m
             X is psd
    Primitives:
        C_op(x) = C x
            R^n --> R^n
        A_op(u) = (Tr(A_1 X), ..., Tr(A_m X))
             = mathcal{A} vec(X)
                where mathcal{A} = [vec(A_1), ..., vec(A_m)]
            R^n --> R^m
        A_star_op(u, z) = A^*(z) u
            = sum_i z_i A_i u
            (R^n, R^m) --> R^n
    Algorithm: 3.1 of https://arxiv.org/pdf/1912.02949.pdf
    Init:
        beta_0 = 1
        K = inf
        X = zeros(n, n)
        y = zeros(m)
    for i in range(T):
        beta = beta_0 sqrt(i + 1)
        eta = 2 / (i + 1)
        q_t = t^{1/4} log n
        (lambda, v) = approxMinEvec(C + A^*(y + beta(AX -b)), q_t)
        X = (1 - eta) X + eta (alpha vv^T)
        y = y + gamma(AX - b)
    inputs:
        C_op: linear operator (see first primitive)
        A_op: linear operator (see second primitive)
        A_star_op: linear operator (see third primitive)
        b: right hand side vector (shape (m))
        T: number of iterations
        m: number of constraints
        n: number of rows of matrix of the standard form sdp
    outputs:
        X: primal solution - (n, n) matrix
        y: dual solution - (m) vector
    """
    t0 = time.time()

    # initialize cgal
    X_init, y_init, z_init = cgal_init(m, n)

    # get proj_K from b (no need for b anymore -- just use proj_K)
    proj_K = b_to_proj_K(b)

    # cgal for loop
    cgal_out = cgal_for_loop(A_op, C_op, A_star_op, proj_K, alpha, norm_A, rescale_obj, rescale_feas,
                             cgal_iters,
                             X_init, y_init, z_init,
                             beta0, y_max,
                             jit=jit,
                             lobpcg_iters=lobpcg_iters,
                             lobpcg_tol=lobpcg_tol,
                             warm_start_v=warm_start_v, 
                             lightweight=lightweight)
    cgal_time = time.time() - t0
    cgal_out['time'] = cgal_time
    return cgal_out

def cgal_for_loop(A_op, C_op, A_star_op, proj_K, alpha, norm_A, rescale_obj, rescale_feas,
                  cgal_iters, X_init, y_init, z_init, beta0, y_max,
                  jit, lobpcg_iters, lobpcg_tol, warm_start_v, lightweight):
    m = y_init.size
    n = X_init.shape[0]

    static_dict = dict(C_op=C_op,
                       A_op=A_op,
                       A_star_op=A_star_op,
                       alpha=alpha,
                       norm_A=norm_A,
                       proj_K=proj_K,
                       rescale_obj=rescale_obj,
                       rescale_feas=rescale_feas,
                       m=m,
                       n=n,
                       beta0=beta0,
                       y_max=y_max,
                       lobpcg_iters=lobpcg_iters,
                       lobpcg_tol=lobpcg_tol,
                       warm_start_v=warm_start_v,
                       lightweight=lightweight)
    partial_cgal_iter = partial(cgal_iteration, static_dict=static_dict)
    obj_vals, infeases = jnp.zeros(cgal_iters), jnp.zeros(cgal_iters)
    X_resids, y_resids = jnp.zeros(cgal_iters), jnp.zeros(cgal_iters)
    lobpcg_steps_mat = jnp.zeros(cgal_iters)
    v_init = jnp.ones((n, 1))
    init_val = X_init, y_init, z_init, obj_vals, infeases, X_resids, y_resids, lobpcg_steps_mat, v_init
    if jit:
        final_val = lax.fori_loop(0, cgal_iters, partial_cgal_iter, init_val)
    else:
        final_val = python_fori_loop(0, cgal_iters, partial_cgal_iter, init_val)
    X, y, z, obj_vals, infeases, X_resids, y_resids, lobpcg_steps, v_final = final_val
    cgal_out = dict(X=X, y=y, obj_vals=obj_vals, infeases=infeases, X_resids=X_resids,
                    y_resids=y_resids, lobpcg_steps=lobpcg_steps)
    return cgal_out


def cgal_iteration(i, init_val, static_dict):
    # unpack static_dict which is meant to not change for an entire problem
    #   i and init_val will change and be passed to/from jax.lax.fori_loop
    C_op, A_op, A_star_op = static_dict['C_op'], static_dict['A_op'], static_dict['A_star_op']
    alpha, norm_A = static_dict['alpha'], static_dict['norm_A']
    rescale_obj, rescale_feas = static_dict['rescale_obj'], static_dict['rescale_feas']
    m, n = static_dict['m'], static_dict['n']
    beta0, y_max = static_dict['beta0'], static_dict['y_max']
    lobpcg_iters, lobpcg_tol = static_dict['lobpcg_iters'], static_dict['lobpcg_tol']
    warm_start_v, lightweight = static_dict['warm_start_v'], static_dict['lightweight']
    proj_K = static_dict['proj_K']

    # unpack init_val
    X, y, z, obj_vals, infeases, X_resids, y_resids, lobpcg_steps_mat, prev_v = init_val
    beta = beta0 * jnp.sqrt(i + 2)
    eta = 2 / (i + 2)

    w = proj_K(z + y / beta)
    a_star_z_fixed = y + beta * (z - w)
    A_star_partial_op = partial(A_star_op, z=jnp.expand_dims(a_star_z_fixed, 1))

    # create new operator as input into lanczos
    def evec_op(u):
        # we take the negative since lobpcg_standard finds the largest evec
        return -C_op(u) - A_star_partial_op(u)

    # get minimum eigenvector
    if warm_start_v:
        lobpcg_out = sparse.linalg.lobpcg_standard(evec_op, prev_v, m=lobpcg_iters, tol=lobpcg_tol)
    else:
        lobpcg_out = sparse.linalg.lobpcg_standard(evec_op, jnp.ones((n, 1)), m=lobpcg_iters, tol=lobpcg_tol)

    # quick check to see why warm start does nothing
    # the sparse.linalg.lobpcg_standard orthonormalizes the initial guess
    # XX = sparse.linalg._orthonormalize(prev_v)
    # print('XX', XX)

    lambd, v, lobpcg_steps = lobpcg_out[0], lobpcg_out[1], lobpcg_out[2]

    # we flip the sign because lobpcg_standard looks for the largest
    #   eigenvalue, eigenvector pair
    lambd = -lambd

    # this will be printed if jit set to false
    print('prev_v', prev_v)
    print('z', z)
    print('lambd', lambd)
    print('lobpcg_steps', lobpcg_steps)

    # update z
    v_alpha = jnp.sqrt(alpha) * v * (lambd < 0)
    v_alpha = v_alpha.squeeze()
    vvT = jnp.outer(v_alpha, v_alpha)
    new_z_dir = A_op(vvT)
    z_next = (1 - eta) * z + eta * new_z_dir

    # calculate primal direction
    H = vvT

    # update primal
    X_next = (1 - eta) * X + eta * H

    # update primal objective
    obj_addition = v_alpha @ C_op(v_alpha)

    # obj_vals[index] takes the previous primal objective
    #   in case i == 0 then we take index = 0, else index = i - 1
    #   if i == 0 then the previous primal objective is zero
    # we update the primal objective without forming the matrix
    index = (i - 1) * (i > 0)
    prev_obj = obj_vals[index] / rescale_obj
    primal_obj = (1 - eta) * prev_obj + eta * obj_addition


    # compute gamma
    gamma_rhs = (alpha ** 2) * beta * norm_A * (eta ** 2)

    # dual update
    # w = jnp.multiply(z_next - proj_K(z_next + y / beta), rescale_feas)
    w = z_next - proj_K(z_next + y / beta)
    # primal_infeas_scaled = jnp.linalg.norm(jnp.multiply(w, rescale_feas))
    w_norm = jnp.linalg.norm(w)
    primal_infeas = jnp.linalg.norm(jnp.multiply(z_next - proj_K(z_next), rescale_feas))

    gamma_raw = gamma_rhs / (w_norm ** 2)
    gamma = jnp.min(jnp.array([gamma_raw, beta0]))

    print('gamma', gamma)

    # update dual solutions with min evec
    #   reject if the new ||y_t|| > K
    y_temp = y + gamma * w
    y_next = y + gamma * w * (jnp.linalg.norm(y_temp) <= y_max)

    # update computationally cheap progress
    # infeases = infeases.at[i].set(primal_infeas * rescale_feas)
    infeases = infeases.at[i].set(primal_infeas)
    lobpcg_steps_mat = lobpcg_steps_mat.at[i].set(lobpcg_steps)
    obj_vals = obj_vals.at[i].set(primal_obj * rescale_obj)

    # compute progress and store it if lightweight is set to False
    if not lightweight:
        # obj_vals = obj_vals.at[i].set(jnp.trace(C_op(X)) * rescale_obj)
        # infeases = infeases.at[i].set(jnp.linalg.norm(A_op(X) - b))
        X_resids = X_resids.at[i].set(jnp.linalg.norm(X - X_next))
        y_resids = y_resids.at[i].set(jnp.linalg.norm(y - y_next))
        

    # update the val for the lax.fori_loop
    val = X_next, y_next, z_next, obj_vals, infeases, X_resids, y_resids, lobpcg_steps_mat, v
    return 


def python_fori_loop(lower, upper, body_fun, init_val):
    """
    this method is meant as a copy of the jax.lax.fori_loop version
        we don't jit this
    used as a comparison and to make sure the jit is helping
    """
    val = init_val
    for i in range(lower, upper):
        val = body_fun(i, val)
    return val


def relative_obj(obj_vals, true_obj):
    """
    given the true objective and an array of objective, this returns the 
        relative objective score
    Note: this has nothing to do with the scaling
    """
    return jnp.abs(true_obj - obj_vals) / (1 + jnp.abs(true_obj))


def relative_infeas(infeases, b):
    """
    given b, a vector or matrix with 2 columns, and an array of infeasibility measures, 
        this returns the relative infeasibility score
    Note: this has nothing to do with the scaling
    """
    if jnp.linalg.norm(b) == np.inf:
        return infeases / jnp.sqrt(b.size)
    return infeases / (1 + jnp.linalg.norm(b))

def proj(input, n, zero_cone_int, nonneg_cone_int, soc_proj_sizes, soc_num_proj, sdp_row_sizes,
         sdp_vector_sizes, sdp_num_proj):
    """
    projects the input onto a cone which is a cartesian product of the zero cone,
        non-negative orthant, many second order cones, and many positive semidefinite cones

    Assumes that the ordering is as follows
    zero, non-negative orthant, second order cone, psd cone
    ============================================================================
    SECOND ORDER CONE
    soc_proj_sizes: list of the sizes of the socp projections needed
    soc_num_proj: list of the number of socp projections needed for each size

    e.g. 50 socp projections of size 3 and 1 projection of size 100 would be
    soc_proj_sizes = [3, 100]
    soc_num_proj = [50, 1]
    ============================================================================
    PSD CONE
    sdp_proj_sizes: list of the sizes of the sdp projections needed
    sdp_vector_sizes: list of the sizes of the sdp projections needed
    soc_num_proj: list of the number of socp projections needed for each size

    e.g. 3 sdp projections of size 10, 10, and 100 would be
    sdp_proj_sizes = [10, 100]
    sdp_vector_sizes = [55, 5050]
    sdp_num_proj = [2, 1]
    """

    nonneg = jnp.clip(input[n + zero_cone_int: n + zero_cone_int + nonneg_cone_int], a_min=0)
    projection = jnp.concatenate([input[:n + zero_cone_int], nonneg])

    # soc setup
    num_soc_blocks = len(soc_proj_sizes)

    # avoiding doing inner product using jax so that we can jit
    soc_total = sum(i[0] * i[1] for i in zip(soc_proj_sizes, soc_num_proj))
    soc_bool = num_soc_blocks > 0

    # sdp setup
    num_sdp_blocks = len(sdp_row_sizes)
    sdp_total = sum(i[0] * i[1] for i in zip(sdp_vector_sizes, sdp_num_proj))
    sdp_bool = num_sdp_blocks > 0

    if soc_bool:
        socp = jnp.zeros(soc_total)
        soc_input = input[n+zero_cone_int+nonneg_cone_int:n +
                          zero_cone_int+nonneg_cone_int + soc_total]

        # iterate over the blocks
        start = 0
        for i in range(num_soc_blocks):
            # calculate the end point
            end = start + soc_proj_sizes[i] * soc_num_proj[i]

            # extract the right soc_input
            curr_soc_input = lax.dynamic_slice(
                soc_input, (start,), (soc_proj_sizes[i] * soc_num_proj[i],))

            # reshape so that we vmap all of the socp projections of the same size together
            curr_soc_input_reshaped = jnp.reshape(
                curr_soc_input, (soc_num_proj[i], soc_proj_sizes[i]))
            curr_soc_out_reshaped = soc_proj_single_batch(curr_soc_input_reshaped)
            curr_socp = jnp.ravel(curr_soc_out_reshaped)

            # place in the correct location in the socp vector
            socp = socp.at[start:end].set(curr_socp)

            # update the start point
            start = end

        projection = jnp.concatenate([projection, socp])
    if sdp_bool:
        sdp_proj = jnp.zeros(sdp_total)
        sdp_input = input[n + zero_cone_int + nonneg_cone_int + soc_total:]

        # iterate over the blocks
        start = 0
        for i in range(num_sdp_blocks):
            # calculate the end point
            end = start + sdp_vector_sizes[i] * sdp_num_proj[i]

            # extract the right sdp_input
            curr_sdp_input = lax.dynamic_slice(
                sdp_input, (start,), (sdp_vector_sizes[i] * sdp_num_proj[i],))

            # reshape so that we vmap all of the sdp projections of the same size together
            curr_sdp_input_reshaped = jnp.reshape(
                curr_sdp_input, (sdp_num_proj[i], sdp_vector_sizes[i]))
            curr_sdp_out_reshaped = sdp_proj_batch(curr_sdp_input_reshaped, sdp_row_sizes[i])
            curr_sdp = jnp.ravel(curr_sdp_out_reshaped)

            # place in the correct location in the sdp vector
            sdp_proj = sdp_proj.at[start:end].set(curr_sdp)

            # update the start point
            start = end

        projection = jnp.concatenate([projection, sdp_proj])
    return projection

def sdp_proj_single(x, n):
    """
    x_proj = argmin_y ||y - x||_2^2
                s.t.   y is psd
    x is a vector with shape (n * (n + 1) / 2)

    we need to pass in n to jit this function
        we could extract dim from x.shape theoretically,
        but we cannot jit a function
        whose values depend on the size of inputs
    """
    # convert vector of size (n * (n + 1) / 2) to matrix of shape (n, n)
    X = unvec_symm(x, n)

    # do the eigendecomposition of X
    evals, evecs = jnp.linalg.eigh(X)

    # clip the negative eigenvalues
    evals_plus = jnp.clip(evals, 0, jnp.inf)

    # put the projection together with non-neg eigenvalues
    X_proj = evecs @ jnp.diag(evals_plus) @ evecs.T

    # vectorize the matrix
    x_proj = vec_symm(X_proj)
    return x_proj


def vec_symm(X, triu_indices=None, factor=jnp.sqrt(2)):
    """Returns a vectorized representation of a symmetric matrix `X`.
    Vectorization (including scaling) as per SCS.
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
    """

    # X = X.copy()
    X *= factor
    X = X.at[jnp.diag_indices(X.shape[0])].set(jnp.diagonal(X) / factor)
    if triu_indices is None:
        col_idx, row_idx = jnp.triu_indices(X.shape[0])
    else:
        col_idx, row_idx = triu_indices
    return X[(row_idx, col_idx)]


def unvec_symm(x, dim, triu_indices=None):
    """Returns a dim-by-dim symmetric matrix corresponding to `x`.
    `x` is a vector of length dim*(dim + 1)/2, corresponding to a symmetric
    matrix; the correspondence is as in SCS.
    X = [ X11 X12 ... X1k
              X21 X22 ... X2k
              ...
              Xk1 Xk2 ... Xkk ],
    where
    vec(X) = (X11, sqrt(2)*X21, ..., sqrt(2)*Xk1, X22, sqrt(2)*X32, ..., Xkk)
    """

    X = jnp.zeros((dim, dim))

    # triu_indices gets indices of upper triangular matrix in row-major order
    if triu_indices is None:
        col_idx, row_idx = jnp.triu_indices(dim)
    else:
        col_idx, row_idx = triu_indices
    z = jnp.zeros(x.size)

    if x.ndim > 1:
        for i in range(x.size):
            z = z.at[i].set(x[i][0, 0])
    else:
        z = x

    X = X.at[(row_idx, col_idx)].set(z)

    X = X + X.T
    X /= jnp.sqrt(2)
    X = X.at[jnp.diag_indices(dim)].set(jnp.diagonal(X) * jnp.sqrt(2) / 2)
    return X

def soc_proj_single(input):
    """
    input is a single vector
        input = (s, y) where y is a vector and s is a scalar
    then we call soc_projection
    """
    # break into scalar and vector parts
    y, s = input[1:], input[0]

    # do the projection
    pi_y, pi_s = soc_projection(y, s)

    # stitch the pieces back together
    return jnp.append(pi_s, pi_y)


def soc_projection(x, s):
    """
    returns the second order cone projection of (x, s)
    (y, t) = Pi_{K}(x, s)
    where K = {y, t | ||y||_2 <= t}

    the second order cone admits a closed form solution

    (y, t) = alpha (x, ||x||_2) if ||x|| >= |s|
             (x, s) if ||x|| <= |s|, s >= 0
             (0, 0) if ||x|| <= |s|, s <= 0

    where alpha = (s + ||x||_2) / (2 ||x||_2)

    case 1: ||x|| >= |s|
    case 2: ||x|| >= |s|
        case 2a: ||x|| >= |s|, s >= 0
        case 2b: ||x|| <= |s|, s <= 0

    """
    x_norm = jnp.linalg.norm(x)

    def case1_soc_proj(x, s):
        # case 1: y_norm >= |s|
        val = (s + x_norm) / (2 * x_norm)
        t = val * x_norm
        y = val * x
        return y, t

    def case2_soc_proj(x, s):
        # case 2: y_norm <= |s|
        # case 2a: s > 0
        def case2a(x, s):
            return x, s

        # case 2b: s < 0
        def case2b(x, s):
            return (0.0*jnp.zeros(x.size), 0.0)
        return lax.cond(s >= 0, case2a, case2b, x, s)
    return lax.cond(x_norm >= jnp.abs(s), case1_soc_proj, case2_soc_proj, x, s)


def create_projection_fn(cones, n):
    """
    cones is a dict with keys
    z: zero cone
    l: non-negative cone
    q: second-order cone
    s: positive semidefinite cone

    n is the size of the variable x in the problem
    min 1/2 x^T P x + c^T x
        s.t. Ax + s = b
             s in K
    This function returns a projection Pi
    which is defined by
    Pi(w) = argmin_v ||w - v||_2^2
                s.t. v in C
    where
    C = {0}^n x K^*
    i.e. the cartesian product of the zero cone of length n and the dual
        cone of K
    For all of the cones we consider, the cones are self-dual
    """
    if 'z' in cones.keys():
        zero_cone = cones['z']
    else:
        zero_cone = 0
    if 'l' in cones.keys():
        nonneg_cone = cones['l']
    else:
        nonneg_cone = 0
    soc = 'q' in cones.keys() and len(cones['q']) > 0
    sdp = 's' in cones.keys() and len(cones['s']) > 0
    if soc:
        soc_cones_array = jnp.array(cones['q'])

        # soc_proj_sizes, soc_num_proj are lists
        # need to convert to list so that the item is not a traced object
        soc_proj_sizes, soc_num_proj = count_num_repeated_elements(soc_cones_array)
    else:
        soc_proj_sizes, soc_num_proj = [], []
    if sdp:
        sdp_cones_array = jnp.array(cones['s'])

        # soc_proj_sizes, soc_num_proj are lists
        # need to convert to list so that the item is not a traced object
        sdp_row_sizes, sdp_num_proj = count_num_repeated_elements(sdp_cones_array)
        sdp_vector_sizes = [int(row_size * (row_size + 1) / 2) for row_size in sdp_row_sizes]
    else:
        sdp_row_sizes, sdp_vector_sizes, sdp_num_proj = [], [], []

    projection = partial(proj,
                         n=n,
                         zero_cone_int=int(zero_cone),
                         nonneg_cone_int=int(nonneg_cone),
                         soc_proj_sizes=soc_proj_sizes,
                         soc_num_proj=soc_num_proj,
                         sdp_row_sizes=sdp_row_sizes,
                         sdp_vector_sizes=sdp_vector_sizes,
                         sdp_num_proj=sdp_num_proj,
                         )
    # return jit(projection)
    return projection


def count_num_repeated_elements(vector):
    """
    given a vector, outputs the frequency in a row

    e.g. vector = [5, 5, 10, 10, 5]

    val_repeated = [5, 10, 5]
    num_repeated = [2, 2, 1]
    """
    m = jnp.r_[True, vector[:-1] != vector[1:], True]
    counts = jnp.diff(jnp.flatnonzero(m))
    unq = vector[m[:-1]]
    out = jnp.c_[unq, counts]

    val_repeated = out[:, 0].tolist()
    num_repeated = out[:, 1].tolist()
    return val_repeated, num_repeated


# provides vmapped versions of the projections for the soc and psd cones
soc_proj_single_batch = vmap(soc_proj_single, in_axes=(0), out_axes=(0))
sdp_proj_batch = vmap(sdp_proj_single, in_axes=(0, None), out_axes=(0))