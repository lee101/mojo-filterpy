from __future__ import annotations

import numpy as np
from scipy.linalg import cholesky


class MerweScaledSigmaPoints:
    def __init__(self, n, alpha, beta, kappa, sqrt_method=None, subtract=None):
        self.n, self.alpha, self.beta, self.kappa = n, alpha, beta, kappa
        self.sqrt = cholesky if sqrt_method is None else sqrt_method
        self.subtract = np.subtract if subtract is None else subtract
        lam = alpha**2 * (n + kappa) - n
        c = 0.5 / (n + lam)
        self.Wm = np.full(2 * n + 1, c)
        self.Wc = np.full(2 * n + 1, c)
        self.Wm[0] = lam / (n + lam)
        self.Wc[0] = self.Wm[0] + (1 - alpha**2 + beta)

    def num_sigmas(self):
        return 2 * self.n + 1

    def sigma_points(self, x, P):
        if self.n != np.size(x):
            raise ValueError(f"expected size(x) {self.n}, but size is {np.size(x)}")
        x = np.asarray([x]) if np.isscalar(x) else np.asarray(x)
        P = np.eye(self.n) * P if np.isscalar(P) else np.atleast_2d(P)
        lam = self.alpha**2 * (self.n + self.kappa) - self.n
        U = self.sqrt((lam + self.n) * P)
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = x
        for k in range(self.n):
            sigmas[k + 1] = self.subtract(x, -U[k])
            sigmas[self.n + k + 1] = self.subtract(x, U[k])
        return sigmas


class JulierSigmaPoints:
    def __init__(self, n, kappa=0.0, sqrt_method=None, subtract=None):
        self.n, self.kappa = n, kappa
        self.sqrt = cholesky if sqrt_method is None else sqrt_method
        self.subtract = np.subtract if subtract is None else subtract
        self.Wm = np.full(2 * n + 1, 0.5 / (n + kappa))
        self.Wm[0] = kappa / (n + kappa)
        self.Wc = self.Wm

    def num_sigmas(self):
        return 2 * self.n + 1

    def sigma_points(self, x, P):
        if self.n != np.size(x):
            raise ValueError(f"expected size(x) {self.n}, but size is {np.size(x)}")
        x = np.asarray([x]) if np.isscalar(x) else np.asarray(x)
        P = np.eye(self.n) * P if np.isscalar(P) else np.atleast_2d(P)
        U = self.sqrt((self.n + self.kappa) * P)
        sigmas = np.zeros((2 * self.n + 1, self.n))
        sigmas[0] = x
        for k in range(self.n):
            sigmas[k + 1] = self.subtract(x, -U[k])
            sigmas[self.n + k + 1] = self.subtract(x, U[k])
        return sigmas
