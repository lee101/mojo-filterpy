from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))

from mojofilterpy.kalman import KalmanFilter
from mojofilterpy._lib import addr, f64, lib

n, m, CYCLES = 16, 6, 2000
rng = np.random.default_rng(4)
kf = KalmanFilter(n, m)
kf.x = rng.normal(size=n)
kf.F = np.eye(n) + rng.normal(scale=0.002, size=(n, n))
kf.H = rng.normal(scale=0.2, size=(m, n))
kf.P = np.eye(n)
kf.Q = np.eye(n) * 1e-4
kf.R = np.eye(m) * 0.2
z = np.zeros(m)


def run():
    for _ in range(CYCLES):
        kf.predict()
        kf.update(z)


run()
best = min((lambda: (lambda s: (run(), time.perf_counter() - s)[1])(time.perf_counter()))() for _ in range(3))
print(f"cycle: {best / CYCLES * 1e6:.1f} us")

pr = cProfile.Profile()
pr.enable()
run()
pr.disable()
s = io.StringIO()
pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18)
for line in s.getvalue().splitlines():
    if line.strip() and ("function calls" in line or "ncalls" in line or "mojofilterpy" in line or "numpy" in line or "{built" in line or "method" in line):
        print(line)
