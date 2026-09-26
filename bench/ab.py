from __future__ import annotations

import ctypes
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "python"))

NEW_LIB = os.environ.get("AB_NEW", os.path.join(ROOT, "dist", "libmojo-filterpy.so"))
OLD_LIB = os.environ.get("AB_OLD", "/tmp/mfp-base/libmojo-filterpy.so")

I = ctypes.c_int64
F = ctypes.c_double

BASE_SIGS = {
    "mfp_predict": ([I] * 10 + [F], None),
    "mfp_linear_update": ([I] * 11, I),
    "mfp_unscented_transform": ([I] * 8, None),
    "mfp_cross_variance": ([I] * 9, None),
    "mfp_ukf_update": ([I] * 10, I),
    "mfp_resample_sorted": ([I] * 4, None),
    "mfp_resample_binary": ([I] * 4, None),
    "mfp_neff": ([I, I], F),
}

NEW_SIGS = dict(BASE_SIGS)
NEW_SIGS["mfp_unscented_transform"] = ([I] * 9, None)
NEW_SIGS["mfp_resample_binary"] = ([I] * 6, None)


def load(path, sigs):
    dll = ctypes.CDLL(path)
    for name, (argtypes, restype) in sigs.items():
        try:
            fn = getattr(dll, name)
        except AttributeError:
            continue
        fn.argtypes = argtypes
        fn.restype = restype
    return dll


ad = lambda a: int(a.ctypes.data)


def interleave(fa, fb, rounds=7, calls=200):
    """Alternate variants so machine-load drift hits both equally."""
    ta, tb = [], []
    fa(); fb()
    for _ in range(rounds):
        s = time.perf_counter()
        for _ in range(calls):
            fa()
        ta.append(time.perf_counter() - s)
        s = time.perf_counter()
        for _ in range(calls):
            fb()
        tb.append(time.perf_counter() - s)
    return min(ta) / calls, min(tb) / calls


def predict_args(n=16):
    rng = np.random.default_rng(4)
    a = dict(
        f=np.ascontiguousarray(np.eye(n) + rng.normal(scale=0.002, size=(n, n))),
        q=np.ascontiguousarray(np.eye(n) * 1e-4),
        b=np.ascontiguousarray(np.empty(1)),
        u=np.ascontiguousarray(np.empty(1)),
        x=np.ascontiguousarray(rng.normal(size=n)),
        cov=np.ascontiguousarray(np.eye(n)),
        xw=np.empty(n), pw=np.empty(2 * n * n), n=n,
    )
    a["addr"] = (ad(a["f"]), ad(a["cov"]), ad(a["f"]), ad(a["q"]), ad(a["b"]),
                 ad(a["u"]), ad(a["xw"]), ad(a["pw"]), n, 0, 1.0)
    return a


def update_args(n=16, m=6):
    rng = np.random.default_rng(4)
    a = dict(
        h=np.ascontiguousarray(rng.normal(scale=0.2, size=(m, n))),
        r=np.ascontiguousarray(np.eye(m) * 0.2),
        y=np.ascontiguousarray(rng.normal(size=m)),
        x=np.ascontiguousarray(rng.normal(size=n)),
        cov=np.ascontiguousarray(np.eye(n)),
        gain=np.empty((n, m)), s=np.empty((m, m)), si=np.empty((m, m)),
        work=np.empty(2 * n * n + 2 * m * m + n * m), n=n, m=m,
    )
    a["addr"] = (ad(a["x"]), ad(a["cov"]), ad(a["h"]), ad(a["r"]), ad(a["y"]),
                 ad(a["gain"]), ad(a["s"]), ad(a["si"]), ad(a["work"]), n, m)
    return a


def ut_args(k=65, n=32):
    rng = np.random.default_rng(9)
    w = rng.random(k)
    w /= w.sum()
    a = dict(
        s=np.ascontiguousarray(rng.normal(size=(k, n))),
        wm=np.ascontiguousarray(w), wc=np.ascontiguousarray(w),
        noise=np.ascontiguousarray(np.eye(n) * 0.01),
        mean=np.empty(n), cov=np.empty((n, n)), dev=np.empty(k * n), k=k, n=n,
    )
    base = (ad(a["s"]), ad(a["wm"]), ad(a["wc"]), ad(a["noise"]), ad(a["mean"]), ad(a["cov"]))
    a["addr_old"] = base + (k, n)
    a["addr_new"] = base + (ad(a["dev"]), k, n)
    return a


def binary_args(n=200_000, buckets=None):
    rng = np.random.default_rng(1)
    w = rng.random(n)
    w /= w.sum()
    cum = np.ascontiguousarray(np.cumsum(w))
    cum[-1] = 1.0
    if buckets is None:
        buckets = max(min(n, 1 << 20), 1)
    a = dict(cum=cum, pos=np.ascontiguousarray(rng.random(n)),
             idx=np.empty(n, dtype=np.int64), table=np.zeros(buckets + 2, dtype=np.int64),
             buckets=buckets, n=n)
    a["addr_old"] = (ad(a["cum"]), ad(a["pos"]), ad(a["idx"]), n)
    a["addr_new"] = (ad(a["cum"]), ad(a["pos"]), ad(a["idx"]), ad(a["table"]), n, buckets)
    return a


def _cmp(label, a, b, keys, tol=1e-12):
    worst = max(float(np.abs(a[k] - b[k]).max()) for k in keys)
    scale = max(float(np.abs(a[k]).max()) for k in keys)
    print(f"{label} parity: max abs diff {worst:.3e} (rel {worst/max(scale,1e-300):.3e})")
    assert worst / max(scale, 1e-300) < tol, f"{label} parity FAILED"


def run(which):
    old, new = load(OLD_LIB, BASE_SIGS), load(NEW_LIB, NEW_SIGS)
    print(f"{'kernel':<32}{'old us':>12}{'new us':>12}{'speedup':>10}")
    if which in ("all", "kf"):
        a = predict_args(); b = predict_args()
        old.mfp_predict(*a["addr"]); new.mfp_predict(*b["addr"])
        _cmp("predict", a, b, ("x", "cov"))
        o, n = interleave(lambda: old.mfp_predict(*a["addr"]),
                          lambda: new.mfp_predict(*b["addr"]))
        print(f"{'mfp_predict 16x16':<32}{o*1e6:12.2f}{n*1e6:12.2f}{o/n:9.2f}x")
        a = update_args(); b = update_args()
        old.mfp_linear_update(*a["addr"]); new.mfp_linear_update(*b["addr"])
        _cmp("linear_update", a, b, ("x", "cov", "gain", "s", "si"))
        o, n = interleave(lambda: old.mfp_linear_update(*a["addr"]),
                          lambda: new.mfp_linear_update(*b["addr"]), calls=500)
        print(f"{'mfp_linear_update 16x6':<32}{o*1e6:12.2f}{n*1e6:12.2f}{o/n:9.2f}x")
    if which in ("all", "ut"):
        a = ut_args(); b = ut_args()
        old.mfp_unscented_transform(*a["addr_old"]); new.mfp_unscented_transform(*b["addr_new"])
        _cmp("unscented_transform", a, b, ("mean", "cov"))
        o, n = interleave(lambda: old.mfp_unscented_transform(*a["addr_old"]),
                          lambda: new.mfp_unscented_transform(*b["addr_new"]), calls=200)
        print(f"{'mfp_unscented_transform 65x32':<32}{o*1e6:12.2f}{n*1e6:12.2f}{o/n:9.2f}x")
    if which in ("all", "bin"):
        a = binary_args(); b = binary_args()
        old.mfp_resample_binary(*a["addr_old"]); new.mfp_resample_binary(*b["addr_new"])
        assert np.array_equal(a["idx"], b["idx"]), "resample indexes differ"
        print("resample_binary parity: indexes bitwise equal")
        o, n = interleave(lambda: old.mfp_resample_binary(*a["addr_old"]),
                          lambda: new.mfp_resample_binary(*b["addr_new"]), rounds=5, calls=3)
        print(f"{'mfp_resample_binary 200k':<32}{o*1e6:12.2f}{n*1e6:12.2f}{o/n:9.2f}x")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "all")
