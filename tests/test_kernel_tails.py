import numpy as np
import pytest

from filterpy.kalman import KalmanFilter as RefKF
from filterpy.kalman import unscented_transform as ref_ut

from mojofilterpy._lib import addr, bucket_count, f64, lib, scratch
from mojofilterpy.kalman import KalmanFilter, unscented_transform


@pytest.mark.parametrize("n", [1, 3, 5, 7, 15, 17, 18, 19, 33])
def test_unscented_transform_block_and_scalar_tails_match_upstream(n):
    rng = np.random.default_rng(100 + n)
    k = 2 * n + 1
    sigmas = rng.normal(size=(k, n))
    weights = rng.random(k)
    weights /= weights.sum()
    noise = np.eye(n) * 0.03
    gx, gp = unscented_transform(sigmas, weights, weights, noise)
    rx, rp = ref_ut(sigmas, weights, weights, noise)
    np.testing.assert_allclose(gx, rx, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(gp, rp, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("k", [1, 2, 5])
def test_unscented_transform_few_sigma_points_match_upstream(k):
    rng = np.random.default_rng(200 + k)
    sigmas = rng.normal(size=(k, 6))
    weights = rng.random(k)
    weights /= weights.sum()
    gx, gp = unscented_transform(sigmas, weights, weights)
    rx, rp = ref_ut(sigmas, weights, weights)
    np.testing.assert_allclose(gx, rx, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(gp, rp, rtol=2e-13, atol=2e-13)


def _kf_pair(n, m, seed):
    rng = np.random.default_rng(seed)
    state = {
        "x": rng.normal(size=(n, 1)),
        "F": np.eye(n) + rng.normal(scale=0.01, size=(n, n)),
        "H": rng.normal(scale=0.3, size=(m, n)),
        "P": np.eye(n) * 2.0,
        "Q": np.eye(n) * 1e-3,
        "R": np.eye(m) * 0.5,
    }
    got, ref = KalmanFilter(n, m), RefKF(n, m)
    for kf in (got, ref):
        for name, value in state.items():
            setattr(kf, name, value)
    return got, ref, rng.normal(size=m)


@pytest.mark.parametrize("n,m", [(1, 1), (3, 1), (5, 2), (7, 3), (17, 5), (18, 4)])
def test_kalman_predict_update_odd_shapes_match_upstream(n, m):
    got, ref, z = _kf_pair(n, m, 300 + n * 10 + m)
    for _ in range(3):
        got.predict()
        ref.predict()
        got.update(z)
        ref.update(z)
    for name in ("x", "P", "K", "S", "SI"):
        np.testing.assert_allclose(
            getattr(got, name), getattr(ref, name), rtol=2e-10, atol=2e-10
        )


def test_kalman_predict_with_control_input_matches_upstream():
    n, m, nu = 5, 2, 3
    got, ref, z = _kf_pair(n, m, 401)
    rng = np.random.default_rng(402)
    b = rng.normal(size=(n, nu))
    u = rng.normal(size=(nu, 1))
    for kf in (got, ref):
        kf.B = b
    got.predict(u=u)
    ref.predict(u=u)
    got.update(z)
    ref.update(z)
    for name in ("x", "P", "K"):
        np.testing.assert_allclose(
            getattr(got, name), getattr(ref, name), rtol=2e-10, atol=2e-10
        )


def _reference_search(cumulative, positions):
    n = cumulative.size
    return np.minimum(np.searchsorted(cumulative, positions, side="right"), n - 1)


def _run_kernel(cumulative, positions):
    n = cumulative.size
    indexes = np.empty(n, dtype=np.int64)
    buckets = bucket_count(n)
    _, table_addr = scratch(buckets + 2, np.int64)
    lib().mfp_resample_binary(
        addr(f64(cumulative)), addr(f64(positions)), addr(indexes),
        table_addr, n, buckets,
    )
    return indexes


@pytest.mark.parametrize("n", [1, 2, 3, 5, 17, 64, 1000, 5000])
def test_bucketed_resample_matches_searchsorted(n):
    rng = np.random.default_rng(500 + n)
    weights = rng.random(n)
    weights /= weights.sum()
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    positions = rng.random(n)
    np.testing.assert_array_equal(
        _run_kernel(cumulative, positions), _reference_search(cumulative, positions)
    )


def test_bucketed_resample_handles_ties_and_zero_weights():
    cumulative = np.array([0.0, 0.0, 0.5, 0.5, 0.5, 1.0])
    positions = np.array([0.0, 0.25, 0.5, 0.75, 0.999999, 1.0])
    np.testing.assert_array_equal(
        _run_kernel(cumulative, positions), _reference_search(cumulative, positions)
    )


def test_bucketed_resample_handles_extreme_positions():
    rng = np.random.default_rng(600)
    n = 2048
    cumulative = np.cumsum(rng.random(n) + 0.5)
    cumulative[-1] = 1.0
    extreme = [0.0, 1.0, 0.999999, 1e-18, 0.5, 0.5000001]
    positions = np.concatenate([extreme, rng.random(n - len(extreme))])
    np.testing.assert_array_equal(
        _run_kernel(cumulative, positions), _reference_search(cumulative, positions)
    )


def test_bucketed_resample_spikes_match_searchsorted():
    rng = np.random.default_rng(700)
    n = 4096
    weights = rng.random(n) ** 8
    weights /= weights.sum()
    cumulative = np.cumsum(weights)
    cumulative[-1] = 1.0
    positions = rng.random(n)
    np.testing.assert_array_equal(
        _run_kernel(cumulative, positions), _reference_search(cumulative, positions)
    )


def test_bucket_count_scales_with_length():
    assert bucket_count(1) == 1
    assert bucket_count(10_000) == 10_000
    assert bucket_count(1 << 30) == 1 << 20
