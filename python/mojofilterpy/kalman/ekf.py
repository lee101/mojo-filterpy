from __future__ import annotations

from copy import deepcopy

import numpy as np

from .kalman_filter import KalmanFilter, _linear_update, _noise, _predict


class ExtendedKalmanFilter(KalmanFilter):
    def predict_x(self, u=0):
        self.x = np.dot(self.F, self.x)
        if self.B is not None and not np.isscalar(self.B):
            self.x += np.dot(self.B, u)

    def predict(self, u=0):
        self.x, self.P = _predict(self.x, self.P, self.F, self.Q, self.B, u)
        self.x_prior, self.P_prior = self.x.copy(), self.P.copy()

    def update(
        self, z, HJacobian, Hx, R=None, args=(), hx_args=(),
        residual=np.subtract,
    ):
        if z is None:
            self.z = np.array([[None] * self.dim_z]).T
            self.x_post, self.P_post = self.x.copy(), self.P.copy()
            return
        args = args if isinstance(args, tuple) else (args,)
        hx_args = hx_args if isinstance(hx_args, tuple) else (hx_args,)
        R = self.R if R is None else _noise(R, self.dim_z)
        if np.isscalar(z) and self.dim_z == 1:
            z = np.asarray([z], dtype=float)
        H = HJacobian(self.x, *args)
        self.y = residual(z, Hx(self.x, *hx_args))
        self.x, self.P, self.K, self.S, self.SI = _linear_update(
            self.x, self.P, H, R, self.y
        )
        self.z = deepcopy(z)
        self.x_post, self.P_post = self.x.copy(), self.P.copy()
        self._invalidate_likelihood()

    def predict_update(self, z, HJacobian, Hx, args=(), hx_args=(), u=0):
        args = args if isinstance(args, tuple) else (args,)
        hx_args = hx_args if isinstance(hx_args, tuple) else (hx_args,)
        H = HJacobian(self.x, *args)
        self.x, self.P = _predict(self.x, self.P, self.F, self.Q, self.B, u)
        self.x_prior, self.P_prior = self.x.copy(), self.P.copy()
        self.y = np.asarray(z) - Hx(self.x, *hx_args)
        self.x, self.P, self.K, self.S, self.SI = _linear_update(
            self.x, self.P, H, self.R, self.y
        )
        self.z = deepcopy(z)
        self.x_post, self.P_post = self.x.copy(), self.P.copy()
        self._invalidate_likelihood()
