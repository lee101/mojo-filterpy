# mojo-filterpy

`mojo-filterpy` is a standalone Mojo port of a tested subset of
[FilterPy](https://github.com/rlabbe/filterpy). It keeps the covered class names,
common method signatures, mutable attributes, and NumPy-facing behavior while
moving dense filtering and resampling loops into one compiled Mojo shared
library. It is not a complete replacement for upstream FilterPy.

The Python package is named `mojofilterpy`, so it can be installed beside the
real `filterpy` package for parity testing or gradual adoption.

## Coverage

Covered:

- `KalmanFilter`: predict, Joseph-form update, steady-state predict/update,
  batch filtering, RTS smoothing, fading-memory alpha, and likelihood statistics
- `ExtendedKalmanFilter`: predict, update, combined predict/update, callback
  arguments, and custom residual functions
- `UnscentedKalmanFilter`: predict, update, batch filtering, RTS smoothing,
  custom state/measurement functions, means, residuals, and inverse functions
- `MerweScaledSigmaPoints`, `JulierSigmaPoints`, and `unscented_transform`
- FilterPy's particle-filter support: residual, stratified, systematic, and
  multinomial resampling, plus effective sample size

Not covered:

- information, square-root, ensemble, fading, IMM, and MMAE filters
- fixed-lag smoothing and correlated-noise Kalman updates
- `SimplexSigmaPoints`
- FilterPy's plotting, discretization, and statistics helper modules

FilterPy 1.4.5 does not provide a stateful particle-filter class; its public
particle-filter API is the `filterpy.monte_carlo` resampling family. This port
covers that family instead of inventing a class that could not be drop-in
compatible.

## Install

The repository pins the tested Mojo nightly and includes FilterPy itself as a
parity-test dependency.

```bash
pixi install
pixi run build
pixi run test
```

For packaging outside the checkout, build the shared library first and point
`MOJOFILTERPY_LIB` at `libmojo-filterpy.so` if it is not in the repository's
`dist/` directory.

## Usage

This complete example is also a useful smoke test:

```python
import numpy as np
from mojofilterpy.kalman import KalmanFilter

kf = KalmanFilter(dim_x=2, dim_z=1)
kf.x = np.array([[0.0], [1.0]])
kf.F = np.array([[1.0, 1.0], [0.0, 1.0]])
kf.H = np.array([[1.0, 0.0]])
kf.P *= 10.0
kf.Q = np.eye(2) * 0.01
kf.R = np.array([[0.25]])

for position in [1.1, 1.9, 3.2, 3.9]:
    kf.predict()
    kf.update(position)

print(kf.x.ravel())
```

Run a saved script inside the environment with `pixi run python script.py`.

## Benchmarks

Measured by `pixi run bench` on this machine. The command reported an Intel(R)
Xeon(R) CPU E5-2697 v4 @ 2.30GHz, Linux 6.8.0-136-generic, Python 3.13.14, and
FilterPy 1.4.5. The benchmark takes the best of five timed runs after one
warm-up. A speedup below 1x means mojo-filterpy was slower.

| Benchmark | mojo-filterpy | FilterPy 1.4.5 | Speedup |
|---|---:|---:|---:|
| KF predict+update, 16x6 (1000 cycles) | 137.20 ms | 102.40 ms | 0.75x |
| Unscented transform, 65x32 (2500 calls) | 172.73 ms | 109.11 ms | 0.63x |
| UKF cross variance, 33x16x8 (4000 calls) | 148.93 ms | 1002.60 ms | 6.73x |
| Systematic resample, 200k (5 calls) | 28.80 ms | 724.25 ms | 25.14x |
| Multinomial resample, 200k (5 calls) | 154.82 ms | 225.30 ms | 1.46x |

These results are a snapshot, not a general speed claim. Re-run
`pixi run bench` on the target machine before making deployment decisions.

## How it works

`src/filterpy.mojo` is one compilation unit exported as
`dist/libmojo-filterpy.so`. Python loads it with `ctypes`. Arrays cross the C ABI
as integer addresses because exported Mojo functions cannot use origin-parametric
pointers. NumPy retains ownership of every input, output, and scratch buffer, so
there is no cross-language allocator or lifetime protocol.

The Python boundary validates shapes, rejects lossy numeric conversion, and
materializes every native argument as a non-empty C-contiguous NumPy buffer.
All numeric buffers are row-major `float64`; resampling indexes are written as
`int64` and converted to FilterPy's public index dtype where required.
The wrapper evaluates arbitrary EKF/UKF callbacks in Python, then sends the
resulting dense matrices and sigma-point batches through Mojo. Linear updates use
the numerically stable Joseph covariance form used by FilterPy.

The test suite installs the real upstream package and asserts numerical and
behavioral parity for filtering trajectories, callbacks, batch filters,
smoothers, sigma points, transforms, likelihoods, and seeded particle
resampling.
