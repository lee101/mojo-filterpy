from __future__ import annotations

import numpy as np

from .._lib import addr, f64, i64, lib


def _weights(weights):
    values = f64(weights)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("weights must be a non-empty one-dimensional array")
    return values


def _sorted_resample(weights, positions):
    weights, positions = _weights(weights), f64(positions)
    if positions.ndim != 1 or positions.size != weights.size:
        raise ValueError("positions must have the same length as weights")
    indexes = np.empty(weights.size, dtype=np.int64)
    lib().mfp_resample_sorted(
        addr(weights), addr(positions), addr(indexes), weights.size
    )
    return indexes


def stratified_resample(weights):
    weights = _weights(weights)
    positions = (np.random.random(weights.size) + np.arange(weights.size)) / weights.size
    return _sorted_resample(weights, positions).astype(np.int32)


def systematic_resample(weights):
    weights = _weights(weights)
    positions = (np.random.random() + np.arange(weights.size)) / weights.size
    return _sorted_resample(weights, positions).astype(np.int32)


def multinomial_resample(weights):
    weights = _weights(weights)
    positions = f64(np.random.random(weights.size))
    indexes = np.empty(weights.size, dtype=np.int64)
    cumulative = f64(np.cumsum(weights))
    cumulative[-1] = 1.0
    lib().mfp_resample_binary(
        addr(cumulative), addr(positions), addr(indexes), weights.size
    )
    return indexes


def residual_resample(weights):
    weights = _weights(weights)
    n = weights.size
    copies = np.floor(n * weights).astype(int)
    deterministic = np.repeat(np.arange(n, dtype=np.int64), copies)
    remaining = n - deterministic.size
    if remaining == 0:
        return deterministic.astype(np.int32)
    residual = weights - copies
    residual /= residual.sum()
    positions = f64(np.random.random(remaining))
    sampled = np.empty(remaining, dtype=np.int64)
    # The binary kernel's length also defines the number of positions, so pad
    # positions and discard the tail while retaining the upstream distribution.
    padded = f64(np.resize(positions, n))
    all_indexes = np.empty(n, dtype=np.int64)
    cumulative = f64(np.cumsum(residual))
    cumulative[-1] = 1.0
    lib().mfp_resample_binary(addr(cumulative), addr(padded), addr(all_indexes), n)
    sampled[:] = all_indexes[:remaining]
    return np.concatenate((deterministic, sampled)).astype(np.int32)


def neff(weights):
    weights = _weights(weights)
    return lib().mfp_neff(addr(weights), weights.size)
