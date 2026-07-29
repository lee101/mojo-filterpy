from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time

import numpy as np

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)

from filterpy.kalman import KalmanFilter as RefKF  # noqa: E402
from filterpy.kalman import MerweScaledSigmaPoints as RefPoints  # noqa: E402
from filterpy.kalman import UnscentedKalmanFilter as RefUKF  # noqa: E402
from filterpy.kalman import unscented_transform as ref_ut  # noqa: E402
from filterpy.monte_carlo import multinomial_resample as ref_multinomial  # noqa: E402
from filterpy.monte_carlo import systematic_resample as ref_systematic  # noqa: E402

from mojofilterpy.kalman import KalmanFilter, MerweScaledSigmaPoints  # noqa: E402
from mojofilterpy.kalman import UnscentedKalmanFilter, unscented_transform  # noqa: E402
from mojofilterpy.monte_carlo import multinomial_resample, systematic_resample  # noqa: E402


def best_time(fn, repeat=5):
    best = math.inf
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def cpu_name():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown CPU"


def kf_case(cls, cycles=1000, n=16, m=6):
    rng = np.random.default_rng(4)
    obj = cls(n, m)
    obj.x = rng.normal(size=n)
    obj.F = np.eye(n) + rng.normal(scale=0.002, size=(n, n))
    obj.H = rng.normal(scale=0.2, size=(m, n))
    obj.P = np.eye(n)
    obj.Q = np.eye(n) * 1e-4
    obj.R = np.eye(m) * 0.2
    z = np.zeros(m)

    def run():
        for _ in range(cycles):
            obj.predict()
            obj.update(z)

    return run


def ukf_objects(nx=16, nz=8):
    def fx(x, dt):
        return x

    def hx(x):
        return x[:nz]

    gp = MerweScaledSigmaPoints(nx, 0.2, 2.0, 0.0)
    rp = RefPoints(nx, 0.2, 2.0, 0.0)
    return (
        UnscentedKalmanFilter(nx, nz, 0.1, hx, fx, gp),
        RefUKF(nx, nz, 0.1, hx, fx, rp),
    )


def cross_case(obj, calls=4000):
    rng = np.random.default_rng(7)
    k, nx, nz = obj._num_sigmas, obj._dim_x, obj._dim_z
    sf = rng.normal(size=(k, nx))
    sh = rng.normal(size=(k, nz))
    x, z = rng.normal(size=nx), rng.normal(size=nz)

    def run():
        for _ in range(calls):
            obj.cross_variance(x, z, sf, sh)

    return run


def ut_case(fn, calls=2500, n=32):
    rng = np.random.default_rng(9)
    sigmas = rng.normal(size=(2 * n + 1, n))
    weights = rng.random(2 * n + 1)
    weights /= weights.sum()
    noise = np.eye(n) * 0.01

    def run():
        for _ in range(calls):
            fn(sigmas, weights, weights, noise)

    return run


def resample_case(fn, weights, calls=5):
    def run():
        for _ in range(calls):
            fn(weights)

    return run


def gpu_free_mib():
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        return min(int(line.split()[0]) for line in result.stdout.splitlines())
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return 0


def main():
    rng = np.random.default_rng(1)
    weights = rng.random(200_000)
    weights /= weights.sum()
    got_ukf, ref_ukf = ukf_objects()
    cases = [
        ("KF predict+update, 16x6 (1000 cycles)", kf_case(KalmanFilter), kf_case(RefKF)),
        ("Unscented transform, 65x32 (2500 calls)", ut_case(unscented_transform), ut_case(ref_ut)),
        ("UKF cross variance, 33x16x8 (4000 calls)", cross_case(got_ukf), cross_case(ref_ukf)),
        ("Systematic resample, 200k (5 calls)", resample_case(systematic_resample, weights), resample_case(ref_systematic, weights)),
        ("Multinomial resample, 200k (5 calls)", resample_case(multinomial_resample, weights), resample_case(ref_multinomial, weights)),
    ]
    print(f"Machine: {cpu_name()}; {platform.system()} {platform.release()}; Python {platform.python_version()}")
    print()
    print("| Benchmark | mojo-filterpy | FilterPy 1.4.5 | Speedup |")
    print("|---|---:|---:|---:|")
    for name, got, ref in cases:
        got()
        ref()
        a, b = best_time(got), best_time(ref)
        print(f"| {name} | {a * 1000:.2f} ms | {b * 1000:.2f} ms | {b / a:.2f}x |")
    if os.environ.get("MOJOFILTERPY_BENCH_GPU") == "1":
        free = gpu_free_mib()
        print()
        if free < 4000:
            print(f"GPU benchmark skipped: only {free} MiB free.")
            return
        n = 256
        cpu = ut_case(
            lambda *args: unscented_transform(*args, device="cpu"), calls=1, n=n
        )
        gpu = ut_case(
            lambda *args: unscented_transform(*args, device="gpu"), calls=1, n=n
        )
        gpu()
        a, b = best_time(cpu, repeat=3), best_time(gpu, repeat=3)
        print("| Optional GPU benchmark | CPU | GPU | GPU speedup |")
        print("|---|---:|---:|---:|")
        print(
            f"| Unscented transform, {2 * n + 1}x{n} (1 call) | "
            f"{a * 1000:.2f} ms | {b * 1000:.2f} ms | {a / b:.2f}x |"
        )


if __name__ == "__main__":
    main()
