from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojofilterpy._lib import addr, lib


def best(fn, repeat=3):
    out = math.inf
    for _ in range(repeat):
        t = time.perf_counter()
        fn()
        out = min(out, time.perf_counter() - t)
    return out


def probe(label, calls, setup, invoke):
    arrays = setup()
    invoke(*arrays)
    return best(lambda: [invoke(*arrays) for _ in range(calls)]) / calls * 1e6


def predict(calls=2000, n=16):
    rng = np.random.default_rng(4)
    return probe("predict", calls, lambda: (
        np.ascontiguousarray(np.eye(n) + rng.normal(scale=0.002, size=(n, n))),
        np.ascontiguousarray(np.eye(n) * 1e-4),
        np.ascontiguousarray(np.empty(1)),
        np.ascontiguousarray(np.empty(1)),
        np.ascontiguousarray(rng.normal(size=n)),
        np.ascontiguousarray(np.eye(n)),
        np.empty(n),
        np.empty((n, n)),
    ), lambda f, q, b, u, x, cov, xw, pw: lib().mfp_predict(
        addr(x), addr(cov), addr(f), addr(q), addr(b), addr(u), addr(xw), addr(pw), n, 0, 1.0))


def linear_update(calls=2000, n=16, m=6):
    rng = np.random.default_rng(4)
    return probe("update", calls, lambda: (
        np.ascontiguousarray(rng.normal(scale=0.2, size=(m, n))),
        np.ascontiguousarray(np.eye(m) * 0.2),
        np.ascontiguousarray(rng.normal(size=m)),
        np.ascontiguousarray(rng.normal(size=n)),
        np.ascontiguousarray(np.eye(n)),
        np.empty((n, m)),
        np.empty((m, m)),
        np.empty((m, m)),
        np.empty(2 * n * n + 2 * m * m + n * m),
    ), lambda h, r, y, x, cov, gain, s, si, work: lib().mfp_linear_update(
        addr(x), addr(cov), addr(h), addr(r), addr(y), addr(gain), addr(s), addr(si), addr(work), n, m))


def ut(calls=2000, k=65, n=32):
    rng = np.random.default_rng(9)
    w = rng.random(k)
    w /= w.sum()
    return probe("ut", calls, lambda: (
        np.ascontiguousarray(rng.normal(size=(k, n))),
        np.ascontiguousarray(w),
        np.ascontiguousarray(w),
        np.ascontiguousarray(np.eye(n) * 0.01),
        np.empty(n),
        np.empty((n, n)),
    ), lambda s, wm, wc, noise, mean, cov: lib().mfp_unscented_transform(
        addr(s), addr(wm), addr(wc), addr(noise), addr(mean), addr(cov), k, n))


def resample_binary(calls=20, n=200_000):
    rng = np.random.default_rng(1)
    w = rng.random(n)
    w /= w.sum()
    cum = np.ascontiguousarray(np.cumsum(w))
    cum[-1] = 1.0
    return probe("resample_binary", calls, lambda: (
        cum,
        np.ascontiguousarray(rng.random(n)),
        np.empty(n, dtype=np.int64),
    ), lambda cum, pos, idx: lib().mfp_resample_binary(addr(cum), addr(pos), addr(idx), n))


def overhead(calls=20000):
    x = np.ascontiguousarray(np.ones(4))
    y = np.ascontiguousarray(np.ones(4))
    idx = np.empty(4, dtype=np.int64)
    fn = lib().mfp_resample_sorted
    return best(lambda: [fn(addr(x), addr(y), addr(idx), 1) for _ in range(calls)]) / calls * 1e6


if __name__ == "__main__":
    print(f"ctypes+addr overhead         {overhead():8.3f} us")
    print(f"mfp_predict 16x16           {predict():8.3f} us")
    print(f"mfp_linear_update 16x6      {linear_update():8.3f} us")
    print(f"mfp_unscented_transform 65x32 {ut():8.3f} us")
    print(f"mfp_resample_binary 200k    {resample_binary():8.3f} us")
