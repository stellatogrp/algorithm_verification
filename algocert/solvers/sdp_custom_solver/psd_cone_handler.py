import numpy as np
import scipy.sparse as spa


class PSDConeHandler(object):

    def __init__(self, ranges):
        self.ranges = ranges  # note the range object should not include a -1, we add it implicitly
        if not isinstance(ranges, list):
            self.ranges = [ranges]
        else:
            self.ranges = ranges

        self._process_ranges()

    def _process_ranges(self):
        debug = True
        if not debug:
            row_indices = []
            ranges_dim = 0
            for r in self.ranges:
                row_indices += list(range(r[0], r[1]))
                ranges_dim += (r[1] - r[0])
            row_indices.append(-1)
            ranges_dim += 1

            self.row_indices = list(set(row_indices))
            # self.ranges_dim = ranges_dim
            self.ranges_dim = len(self.row_indices)
        else:
            row_indices = []
            used_indices = set([])
            for r in self.ranges:
                curr_indices = list(range(r[0], r[1]))
                for c in curr_indices:
                    if c in used_indices:
                        continue
                    row_indices.append(c)
                    used_indices.add(c)
            row_indices.sort()
            self.row_indices = row_indices + [-1]
            self.ranges_dim = len(self.row_indices)

    def get_H_mat(self, n):
        '''
            probably wont use this, corresponds to when vec(X) is the entire stack
                of dimension n ** 2 instead of n(n+1) / 2
        '''
        E = self.get_E_mat(n)
        return np.kron(E, E)

    def get_sparse_Hsymm_mat(self, n):
        return spa.csc_matrix(self.get_Hsymm_mat(n))

    def get_Hsymm_mat(self, n):
        # r = chordal_vec.size
        # E = jnp.zeros((r, n))
        # E = E.at[jnp.arange(r), chordal_vec].set(1)
        E = self.get_E_mat(n)
        r = E.shape[0]
        H = np.kron(E, E)
        a = np.arange(n ** 2).reshape(n, n)
        # a[np.diag_indices(n)]
        triu_n_cols = a[np.triu_indices(n)]

        # H_symm = H / np.sqrt(2)
        H_symm = H
        # H_diag_col_vals = H_symm[:, diag_n_cols]
        # H_symm[:, diag_n_cols] = H_diag_col_vals * np.sqrt(2)
        # H_symm[:, diag_n_cols] = H_diag_col_vals

        b = np.arange(r ** 2).reshape(r, r)
        b[np.diag_indices(r)]
        triu_r_rows = b[np.triu_indices(r)]

        # H_symm = H_symm * np.sqrt(2)
        # H_diag_row_vals = H_symm[diag_r_rows, :]
        # H_symm[diag_r_rows, :] = H_diag_row_vals / np.sqrt(2)
        # H_symm[diag_r_rows, :] = H_diag_row_vals

        H_symm = H_symm[triu_r_rows[:, np.newaxis], triu_n_cols]

        # print(H_symm)
        # exit(0)
        return H_symm

    def get_E_mat(self, n):
        r = self.ranges_dim
        E = np.zeros((r, n))
        E[np.arange(r), self.row_indices] = 1
        return E


# The vec function as documented in api/cones
def vec(S):
    n = S.shape[0]
    S = np.copy(S)
    S *= np.sqrt(2)
    S[range(n), range(n)] /= np.sqrt(2)
    return S[np.triu_indices(n)]


# The mat function as documented in api/cones
def mat(s):
    n = int((np.sqrt(8 * len(s) + 1) - 1) / 2)
    S = np.zeros((n, n))
    S[np.triu_indices(n)] = s / np.sqrt(2)
    S = S + S.T
    S[range(n), range(n)] /= np.sqrt(2)
    return S

# def vec_symm_kron(E):
#     """
#     given a matrix E, this creates a matrix H s.t.
#         Y = E X E.T
#         vec_symm(Y) = H_symm vec_symm(X)

#     E has shape (r, n)
#     H_symm will have shape (r * (r + 1) / 2, n * (n + 1) / 2)
#     """
#     r, n = E.shape
#     H = jnp.kron(E, E)

#     # get column indices that correspond to diagonals, upper triangular
#     a = np.arange(n ** 2).reshape(n, n)
#     diag_n_cols = a[jnp.diag_indices(n)]
#     triu_n_cols = a[jnp.triu_indices(n)]

#     # divide non-diagonal columns by sqrt(2)
#     H_symm = H / jnp.sqrt(2)
#     H_diag_col_vals = H_symm[:, diag_n_cols]
#     H_symm = H_symm.at[:, diag_n_cols].set(H_diag_col_vals * jnp.sqrt(2))

#     # get row indices that correspond to diagonals, upper triangular
#     b = np.arange(r ** 2).reshape(r, r)
#     diag_r_rows = b[jnp.diag_indices(r)]
#     triu_r_rows = b[jnp.triu_indices(r)]

#     # multiply non-diagonal rows by sqrt(2)
#     H_symm = H_symm * jnp.sqrt(2)
#     H_diag_row_vals = H_symm[diag_r_rows, :]
#     H_symm = H_symm.at[diag_r_rows, :].set(H_diag_row_vals / jnp.sqrt(2))

#     # cut the lower triangular rows and cols
#     H_symm = H_symm[triu_r_rows[:, jnp.newaxis], triu_n_cols]
#     return H_symm


def main():
    ranges = [(3, 4), (0, 2)]
    # ranges = [(0, 4)]
    n = 5
    h = PSDConeHandler(ranges)
    E = h.get_E_mat(n)
    print(E)

    X = np.array([
        [1, 2, 3, 4, 5],
        [2, 6, 7, 8, 9],
        [3, 7, 10, 11, 12],
        [4, 8, 11, 13, 14],
        [5, 9, 12, 14, 15]
    ], dtype='float64')

    print(E @ X @ E.T)

    print('----')
    print(vec(X), vec(X).shape)
    print(spa.csc_matrix(vec(X)))
    print(mat(vec(X)))

    print('----')
    H_symm = h.get_Hsymm_mat(n)
    print(H_symm)
    print(h.get_sparse_Hsymm_mat(n).todense())
    print(mat(H_symm @ vec(X)))

if __name__ == '__main__':
    main()
