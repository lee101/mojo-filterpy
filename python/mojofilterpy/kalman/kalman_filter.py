from __future__ import annotations

from copy import deepcopy

import numpy as np

from .._lib import addr, f64, lib, scratch
from .._stats import LikelihoodMixin


def _noise(value, n: int) -> np.ndarray:
    if type(value) is np.ndarray:
        return f64(value)
    if np.isscalar(value):
        return np.eye(n) * float(value)
    return f64(value)


def _reshape_z(z, dim_z: int, ndim: int):
    z = np.asarray(z, dtype=float)
    if z.size != dim_z:
        raise ValueError(f"z must be convertible to shape ({dim_z}, 1)")
    return z.reshape(dim_z) if ndim == 1 else z.reshape(dim_z, 1)


_DUMMY = np.empty(1, dtype=np.float64)
_DUMMY_ADDR = int(_DUMMY.ctypes.data)


def _predict(x, P, F, Q, B, u, alpha_sq=1.0):
    source = np.asarray(x)
    shape = source.shape
    xv = f64(source, copy=True)
    n = xv.size
    if n == 0:
        raise ValueError("state must be non-empty")
    if xv.ndim != 1:
        xv = xv.reshape(n)
    cov = f64(P, copy=True)
    if cov.shape != (n, n):
        cov = cov.reshape(n, n)
    fm = f64(F)
    if fm.shape != (n, n):
        fm = fm.reshape(n, n)
    qm = f64(_noise(Q, n))
    if qm.shape != (n, n):
        qm = qm.reshape(n, n)
    nu = 0
    bm = uv = None
    bm_addr = uv_addr = _DUMMY_ADDR
    if B is not None and u is not None and not np.isscalar(B):
        uv = f64(u)
        if uv.ndim != 1:
            uv = uv.reshape(-1)
        nu = uv.size
        bm = f64(B)
        if bm.shape != (n, nu):
            bm = bm.reshape(n, nu)
        bm_addr, uv_addr = addr(bm), addr(uv)
    _, xw_addr = scratch(n)
    _, work_addr = scratch(2 * n * n)
    lib().mfp_predict(
        addr(xv), addr(cov), addr(fm), addr(qm), bm_addr, uv_addr,
        xw_addr, work_addr, n, nu, alpha_sq,
    )
    return xv if xv.shape == shape else xv.reshape(shape), cov


def _linear_update(x, P, H, R, y):
    source = np.asarray(x)
    shape = source.shape
    xv = f64(source, copy=True)
    n = xv.size
    if n == 0:
        raise ValueError("state must be non-empty")
    if xv.ndim != 1:
        xv = xv.reshape(n)
    cov = f64(P, copy=True)
    if cov.shape != (n, n):
        cov = cov.reshape(n, n)
    hm = f64(H)
    if hm.ndim != 2 or hm.shape[0] == 0:
        raise ValueError("H must be a non-empty two-dimensional matrix")
    m = hm.shape[0]
    if hm.shape != (m, n):
        hm = hm.reshape(m, n)
    rm = f64(_noise(R, m))
    if rm.shape != (m, m):
        rm = rm.reshape(m, m)
    yv = f64(y)
    if yv.ndim != 1:
        yv = yv.reshape(m)
    gain = np.empty((n, m), dtype=np.float64)
    s = np.empty((m, m), dtype=np.float64)
    si = np.empty((m, m), dtype=np.float64)
    _, work_addr = scratch(2 * n * n + 2 * m * m + n * m)
    ok = lib().mfp_linear_update(
        addr(xv), addr(cov), addr(hm), addr(rm), addr(yv), addr(gain),
        addr(s), addr(si), work_addr, n, m,
    )
    if not ok:
        raise np.linalg.LinAlgError("innovation covariance is singular")
    return (xv if xv.shape == shape else xv.reshape(shape)), cov, gain, s, si


class KalmanFilter(LikelihoodMixin):
    def __init__(self, dim_x, dim_z, dim_u=0):
        if dim_x < 1 or dim_z < 1 or dim_u < 0:
            raise ValueError("dimensions must be positive (dim_u may be zero)")
        self.dim_x, self.dim_z, self.dim_u = dim_x, dim_z, dim_u
        self.x = np.zeros((dim_x, 1))
        self.P = np.eye(dim_x)
        self.B = 0
        self.F = np.eye(dim_x)
        self.H = np.zeros((dim_z, dim_x))
        self.R = np.eye(dim_z)
        self.Q = np.eye(dim_x)
        self.M = np.zeros((dim_x, dim_z))
        self.z = np.array([[None] * dim_z]).T
        self.K = np.zeros((dim_x, dim_z))
        self.y = np.zeros((dim_z, 1))
        self.S = np.zeros((dim_z, dim_z))
        self.SI = np.zeros((dim_z, dim_z))
        self._I = np.eye(dim_x)
        self._alpha_sq = 1.0
        self.inv = np.linalg.inv
        self.x_prior = self.x.copy()
        self.P_prior = self.P.copy()
        self.x_post = self.x.copy()
        self.P_post = self.P.copy()
        self._invalidate_likelihood()

    @property
    def alpha(self):
        return np.sqrt(self._alpha_sq)

    @alpha.setter
    def alpha(self, value):
        if value <= 0:
            raise ValueError("alpha must be greater than 0")
        self._alpha_sq = value**2

    def predict(self, u=None, B=None, F=None, Q=None):
        B = self.B if B is None else B
        F = self.F if F is None else F
        Q = self.Q if Q is None else Q
        self.x, self.P = _predict(self.x, self.P, F, Q, B, u, self._alpha_sq)
        self.x_prior, self.P_prior = self.x.copy(), self.P.copy()

    def update(self, z, R=None, H=None):
        self._invalidate_likelihood()
        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post, self.P_post = self.x.copy(), self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return
        H = self.H if H is None else H
        R = self.R if R is None else R
        z = _reshape_z(z, self.dim_z, np.asarray(self.x).ndim)
        self.y = z - np.dot(H, self.x)
        if self.inv is np.linalg.inv:
            self.x, self.P, self.K, self.S, self.SI = _linear_update(
                self.x, self.P, H, R, self.y
            )
        else:
            R = _noise(R, self.dim_z)
            pht = self.P @ np.asarray(H).T
            self.S = np.asarray(H) @ pht + R
            self.SI = self.inv(self.S)
            self.K = pht @ self.SI
            self.x = self.x + self.K @ self.y
            i_kh = self._I - self.K @ H
            self.P = i_kh @ self.P @ i_kh.T + self.K @ R @ self.K.T
        self.z = deepcopy(z)
        self.x_post, self.P_post = self.x.copy(), self.P.copy()

    def predict_steadystate(self, u=0, B=None):
        B = self.B if B is None else B
        self.x = np.dot(self.F, self.x)
        if B is not None and not np.isscalar(B):
            self.x += np.dot(B, u)
        self.x_prior, self.P_prior = self.x.copy(), self.P.copy()

    def update_steadystate(self, z):
        self._invalidate_likelihood()
        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post, self.P_post = self.x.copy(), self.P.copy()
            self.y = np.zeros((self.dim_z, 1))
            return
        z = _reshape_z(z, self.dim_z, np.asarray(self.x).ndim)
        self.y = z - np.dot(self.H, self.x)
        self.x = self.x + np.dot(self.K, self.y)
        self.z = deepcopy(z)
        self.x_post, self.P_post = self.x.copy(), self.P.copy()

    def batch_filter(
        self, zs, Fs=None, Qs=None, Hs=None, Rs=None, Bs=None, us=None,
        update_first=False, saver=None,
    ):
        n = len(zs)
        Fs = [self.F] * n if Fs is None else Fs
        Qs = [self.Q] * n if Qs is None else Qs
        Hs = [self.H] * n if Hs is None else Hs
        Rs = [self.R] * n if Rs is None else Rs
        Bs = [self.B] * n if Bs is None else Bs
        us = [0] * n if us is None else us
        means = np.zeros((n,) + self.x.shape)
        means_p = np.zeros_like(means)
        covs = np.zeros((n, self.dim_x, self.dim_x))
        covs_p = np.zeros_like(covs)
        for i, z in enumerate(zs):
            if update_first:
                self.update(z, Rs[i], Hs[i])
                means[i], covs[i] = self.x, self.P
                self.predict(us[i], Bs[i], Fs[i], Qs[i])
                means_p[i], covs_p[i] = self.x, self.P
            else:
                self.predict(us[i], Bs[i], Fs[i], Qs[i])
                means_p[i], covs_p[i] = self.x, self.P
                self.update(z, Rs[i], Hs[i])
                means[i], covs[i] = self.x, self.P
            if saver is not None:
                saver.save()
        return means, covs, means_p, covs_p

    def rts_smoother(self, Xs, Ps, Fs=None, Qs=None, inv=np.linalg.inv):
        Xs, Ps = np.asarray(Xs), np.asarray(Ps)
        n = len(Xs)
        Fs = [self.F] * n if Fs is None else Fs
        Qs = [self.Q] * n if Qs is None else Qs
        x, P = Xs.copy(), Ps.copy()
        K = np.zeros((n, self.dim_x, self.dim_x))
        Pp = P.copy()
        for k in range(n - 2, -1, -1):
            Pp[k] = Fs[k + 1] @ P[k] @ Fs[k + 1].T + Qs[k + 1]
            K[k] = P[k] @ Fs[k + 1].T @ inv(Pp[k])
            x[k] += K[k] @ (x[k + 1] - Fs[k + 1] @ x[k])
            P[k] += K[k] @ (P[k + 1] - Pp[k]) @ K[k].T
        return x, P, K, Pp
