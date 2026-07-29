from __future__ import annotations

from copy import deepcopy

import numpy as np

from .._lib import addr, f64, lib
from .._stats import LikelihoodMixin


def unscented_transform(
    sigmas, Wm, Wc, noise_cov=None, mean_fn=None, residual_fn=None, device="cpu"
):
    if device not in ("cpu", "gpu"):
        raise ValueError("device must be 'cpu' or 'gpu'")
    sigmas = f64(sigmas)
    if sigmas.ndim != 2 or 0 in sigmas.shape:
        raise ValueError("sigmas must be a non-empty two-dimensional array")
    k, n = sigmas.shape
    Wm, Wc = f64(Wm).reshape(-1), f64(Wc).reshape(-1)
    if Wm.size != k or Wc.size != k:
        raise ValueError("Wm and Wc must have one value per sigma point")
    if mean_fn is not None or residual_fn not in (None, np.subtract):
        mean = np.dot(Wm, sigmas) if mean_fn is None else mean_fn(sigmas, Wm)
        cov = np.zeros((n, n))
        for i in range(k):
            y = sigmas[i] - mean if residual_fn is None else residual_fn(sigmas[i], mean)
            cov += Wc[i] * np.outer(y, y)
        if noise_cov is not None:
            cov += noise_cov
        return mean, cov
    noise = (
        np.zeros((n, n), dtype=np.float64)
        if noise_cov is None
        else f64(noise_cov).reshape(n, n)
    )
    mean = np.empty(n, dtype=np.float64)
    cov = np.empty((n, n), dtype=np.float64)
    use_gpu = device == "gpu" and k * n * n >= 4_000_000
    kernel = (
        lib().mfp_unscented_transform_gpu
        if use_gpu
        else lib().mfp_unscented_transform
    )
    status = kernel(
        addr(sigmas), addr(Wm), addr(Wc), addr(noise), addr(mean), addr(cov), k, n
    )
    if use_gpu and status < 0:
        raise RuntimeError("Mojo GPU unscented transform failed")
    return mean, cov


class UnscentedKalmanFilter(LikelihoodMixin):
    def __init__(
        self, dim_x, dim_z, dt, hx, fx, points, sqrt_fn=None,
        x_mean_fn=None, z_mean_fn=None, residual_x=None, residual_z=None,
    ):
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        self.Q = np.eye(dim_x)
        self.R = np.eye(dim_z)
        self._dim_x, self._dim_z = dim_x, dim_z
        self.points_fn, self._dt = points, dt
        self._num_sigmas = points.num_sigmas()
        self.hx, self.fx = hx, fx
        self.x_mean, self.z_mean = x_mean_fn, z_mean_fn
        self.residual_x = np.subtract if residual_x is None else residual_x
        self.residual_z = np.subtract if residual_z is None else residual_z
        self.Wm, self.Wc = points.Wm, points.Wc
        self.msqrt = np.linalg.cholesky if sqrt_fn is None else sqrt_fn
        self.inv = np.linalg.inv
        self.K = np.zeros((dim_x, dim_z))
        self.y = np.zeros(dim_z)
        self.z = np.array([[None] * dim_z]).T
        self.S = np.zeros((dim_z, dim_z))
        self.SI = np.zeros((dim_z, dim_z))
        self.sigmas_f = np.zeros((self._num_sigmas, dim_x))
        self.sigmas_h = np.zeros((self._num_sigmas, dim_z))
        self.x_prior, self.P_prior = self.x.copy(), self.P.copy()
        self.x_post, self.P_post = self.x.copy(), self.P.copy()
        self._invalidate_likelihood()

    def compute_process_sigmas(self, dt, fx=None, **fx_args):
        fx = self.fx if fx is None else fx
        sigmas = self.points_fn.sigma_points(self.x, self.P)
        for i, sigma in enumerate(sigmas):
            self.sigmas_f[i] = fx(sigma, dt, **fx_args)

    def predict(self, dt=None, UT=None, fx=None, **fx_args):
        dt = self._dt if dt is None else dt
        transform = unscented_transform if UT is None else UT
        self.compute_process_sigmas(dt, fx, **fx_args)
        self.x, self.P = transform(
            self.sigmas_f, self.Wm, self.Wc, self.Q,
            self.x_mean, self.residual_x,
        )
        self.x_prior, self.P_prior = self.x.copy(), self.P.copy()

    def cross_variance(self, x, z, sigmas_f, sigmas_h):
        sf, sh = f64(sigmas_f), f64(sigmas_h)
        if sf.ndim != 2 or sh.ndim != 2 or sf.shape[0] != sh.shape[0]:
            raise ValueError("sigma arrays must be two-dimensional with equal row counts")
        nx, nz = sf.shape[1], sh.shape[1]
        if self.residual_x is not np.subtract or self.residual_z is not np.subtract:
            cross = np.zeros((nx, nz))
            for i in range(sf.shape[0]):
                cross += self.Wc[i] * np.outer(
                    self.residual_x(sf[i], x), self.residual_z(sh[i], z)
                )
            return cross
        xv = f64(x).reshape(nx)
        zv = f64(z).reshape(nz)
        wc = f64(self.Wc).reshape(sf.shape[0])
        cross = np.empty((nx, nz), dtype=np.float64)
        lib().mfp_cross_variance(
            addr(sf), addr(xv), addr(sh), addr(zv), addr(wc), addr(cross),
            sf.shape[0], nx, nz,
        )
        return cross

    def update(self, z, R=None, UT=None, hx=None, **hx_args):
        if z is None:
            self.z = np.array([[None] * self._dim_z]).T
            self.x_post, self.P_post = self.x.copy(), self.P.copy()
            return
        hx = self.hx if hx is None else hx
        transform = unscented_transform if UT is None else UT
        R = self.R if R is None else (
            np.eye(self._dim_z) * R if np.isscalar(R) else R
        )
        self.sigmas_h = np.atleast_2d([hx(s, **hx_args) for s in self.sigmas_f])
        zp, self.S = transform(
            self.sigmas_h, self.Wm, self.Wc, R,
            self.z_mean, self.residual_z,
        )
        pxz = self.cross_variance(self.x, zp, self.sigmas_f, self.sigmas_h)
        self.y = np.asarray(self.residual_z(z, zp), dtype=np.float64)
        xv, cov = f64(self.x, copy=True), f64(self.P, copy=True)
        pxz, sm = f64(pxz), f64(self.S, copy=True)
        yv = f64(self.y).reshape(self._dim_z)
        gain = np.empty((self._dim_x, self._dim_z), dtype=np.float64)
        si = np.empty_like(sm)
        work = np.empty(self._dim_z**2, dtype=np.float64)
        if self.inv is np.linalg.inv:
            ok = lib().mfp_ukf_update(
                addr(xv), addr(cov), addr(pxz), addr(sm), addr(yv),
                addr(gain), addr(si), addr(work), self._dim_x, self._dim_z,
            )
            if not ok:
                raise np.linalg.LinAlgError("innovation covariance is singular")
        else:
            si[:] = self.inv(sm)
            gain[:] = pxz @ si
            xv += gain @ self.y
            cov -= gain @ sm @ gain.T
        self.x, self.P, self.K, self.SI = xv, cov, gain, si
        self.z = deepcopy(z)
        self.x_post, self.P_post = self.x.copy(), self.P.copy()
        self._invalidate_likelihood()

    def batch_filter(self, zs, Rs=None, dts=None, UT=None, saver=None):
        n = len(zs)
        Rs = [self.R] * n if Rs is None else Rs
        dts = [self._dt] * n if dts is None else (
            [dts] * n if np.isscalar(dts) else dts
        )
        means = np.zeros((n,) + self.x.shape)
        covs = np.zeros((n, self._dim_x, self._dim_x))
        for i, (z, r, dt) in enumerate(zip(zs, Rs, dts)):
            self.predict(dt=dt, UT=UT)
            self.update(z, r, UT=UT)
            means[i], covs[i] = self.x, self.P
            if saver is not None:
                saver.save()
        return means, covs

    def rts_smoother(self, Xs, Ps, Qs=None, dts=None, UT=None):
        if len(Xs) != len(Ps):
            raise ValueError("Xs and Ps must have the same length")
        xs, ps = np.asarray(Xs).copy(), np.asarray(Ps).copy()
        n, dim_x = xs.shape
        dts = [self._dt] * n if dts is None else (
            [dts] * n if np.isscalar(dts) else dts
        )
        Qs = [self.Q] * n if Qs is None else Qs
        transform = unscented_transform if UT is None else UT
        gains = np.zeros((n, dim_x, dim_x))
        propagated = np.zeros((self._num_sigmas, dim_x))
        for k in range(n - 2, -1, -1):
            sigmas = self.points_fn.sigma_points(xs[k], ps[k])
            for i in range(self._num_sigmas):
                propagated[i] = self.fx(sigmas[i], dts[k])
            xb, pb = transform(
                propagated, self.Wm, self.Wc, Qs[k],
                self.x_mean, self.residual_x,
            )
            pxb = np.zeros((dim_x, dim_x))
            for i in range(self._num_sigmas):
                pxb += self.Wc[i] * np.outer(
                    self.residual_x(sigmas[i], xs[k]),
                    self.residual_x(propagated[i], xb),
                )
            gain = pxb @ self.inv(pb)
            xs[k] += gain @ self.residual_x(xs[k + 1], xb)
            ps[k] += gain @ (ps[k + 1] - pb) @ gain.T
            gains[k] = gain
        return xs, ps, gains
