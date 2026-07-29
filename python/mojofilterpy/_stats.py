from __future__ import annotations

import sys

import numpy as np


class LikelihoodMixin:
    _log_likelihood: float | None
    _likelihood: float | None
    _mahalanobis: float | None

    def _invalidate_likelihood(self) -> None:
        self._log_likelihood = None
        self._likelihood = None
        self._mahalanobis = None

    @property
    def log_likelihood(self) -> float:
        if self._log_likelihood is None:
            y = np.asarray(self.y, dtype=float).reshape(-1)
            s = np.asarray(self.S, dtype=float)
            sign, logdet = np.linalg.slogdet(s)
            if sign <= 0:
                self._log_likelihood = -np.inf
            else:
                self._log_likelihood = float(
                    -0.5
                    * (
                        y.size * np.log(2.0 * np.pi)
                        + logdet
                        + y @ np.linalg.solve(s, y)
                    )
                )
        return self._log_likelihood

    @property
    def likelihood(self) -> float:
        if self._likelihood is None:
            self._likelihood = max(float(np.exp(self.log_likelihood)), sys.float_info.min)
        return self._likelihood

    @property
    def mahalanobis(self) -> float:
        if self._mahalanobis is None:
            y = np.asarray(self.y, dtype=float).reshape(-1)
            self._mahalanobis = float(np.sqrt(y @ np.linalg.solve(self.S, y)))
        return self._mahalanobis
