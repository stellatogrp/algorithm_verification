import numpy as np


class NymstromSketch(object):
    """
        Represents the Nymstrom Sketch of a matrix S = X @ Omega
        Omega is dimension n x R, where R is the sketch size and (usually) a small number
            (i.e. R is on the order of 100 or so)
        Omega is sampled elementwise from a standard Gaussian
    """

    def __init__(self, n, R, X_init=None, eps=1e-6):
        assert n > R, 'R is too big, needs to be strictly smaller than n'
        self.n = n
        self.R = R
        self.eps = eps
        self.Omega = np.random.randn(n, R)
        if X_init is None:
            self.S = np.zeros((n, R))
        else:
            self.S = X_init @ self.Omega

    def rank_one_update(self, v, eta):
        assert len(v.shape) == 2, f'v shape: {v.shape}'
        assert v.shape[1] == 1, f'v shape: {v.shape}, needs to be n x 1'
        self.S = (1 - eta) * self.S + eta * v @ v.T @ self.Omega

    def reconstruct(self):
        """
            Numerically stable reconstruction, python implementation of
                https://github.com/alpyurtsever/SketchyCGAL/blob/master/solver/%40NystromSketch/NystromSketch.m
        """
        eta = np.sqrt(self.n) * self.eps * np.max(np.linalg.norm(self.S, axis=0))
        S_eta = self.S + eta * self.Omega
        B = self.Omega.T @ S_eta
        B = .5 * (B + B.T)  # this is for robustness in the cholesky factorization

        # source to convert the matlab code:
        #  https://stackoverflow.com/questions/1007442/mrdivide-function-in-matlab-what-is-it-doing-and-how-can-i-do-it-in-python
        # C = np.linalg.cholesky(B).T  # need to be upper triangular, not lower
        C = np.linalg.cholesky(B)
        Y = np.linalg.solve(C, S_eta.T).T  # equivalent to matlab's Y = S_sigma / C.T
        U, Sigma, _ = np.linalg.svd(Y, full_matrices=False)
        # print(Sigma.shape)
        # Sigma_sq = np.diag(np.square(Sigma) - eta)
        # Delta = np.maximum(np.diag(np.square(Sigma) - eta), 0)
        Delta = np.maximum(np.square(Sigma) - eta, 0)
        return U, Delta


def main():
    np.random.seed(0)
    n = 100
    v = np.random.randn(n, 1) + 3
    w = np.random.randn(n, 1)

    R = 25
    # A = np.random.randn(n, n) + 10
    # A = .5 * (A + A.T)
    A = .75 * v @ v.T + .25 * w @ w.T
    # S = NymstromSketch(n, R, X_init=A)

    S = NymstromSketch(n, R)
    S.rank_one_update(v, 1)
    S.rank_one_update(w, .25)
    U, Delta = S.reconstruct()
    print(U.shape, Delta.shape)
    test = U @ np.diag(Delta) @ U.T
    print(np.linalg.norm(A - test))


if __name__ == '__main__':
    main()
