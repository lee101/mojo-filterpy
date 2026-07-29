import numpy as np
import pytest

from filterpy.kalman import JulierSigmaPoints as RefJulier
from filterpy.kalman import MerweScaledSigmaPoints as RefMerwe
from filterpy.kalman import UnscentedKalmanFilter as RefUKF
from filterpy.kalman import unscented_transform as ref_ut

from mojofilterpy.kalman import (
    JulierSigmaPoints,
    MerweScaledSigmaPoints,
    UnscentedKalmanFilter,
    unscented_transform,
)


def test_merwe_weights_and_sigma_points_match():
    got, ref = MerweScaledSigmaPoints(3, 0.2, 2.0, -1.0), RefMerwe(3, 0.2, 2.0, -1.0)
    x = np.array([1.0, -2.0, 0.5])
    P = np.array([[2.0, 0.1, 0.2], [0.1, 1.0, 0.05], [0.2, 0.05, 0.7]])
    np.testing.assert_allclose(got.Wm, ref.Wm)
    np.testing.assert_allclose(got.Wc, ref.Wc)
    np.testing.assert_allclose(got.sigma_points(x, P), ref.sigma_points(x, P))
    assert got.num_sigmas() == ref.num_sigmas()


def test_merwe_scalar_sigma_points_match():
    got, ref = MerweScaledSigmaPoints(1, 0.3, 2.0, 0.0), RefMerwe(1, 0.3, 2.0, 0.0)
    np.testing.assert_allclose(got.sigma_points(4.0, 2.0), ref.sigma_points(4.0, 2.0))


def test_julier_weights_and_sigma_points_match():
    got, ref = JulierSigmaPoints(2, 1.0), RefJulier(2, 1.0)
    P = np.array([[1.2, 0.2], [0.2, 0.9]])
    np.testing.assert_allclose(got.Wm, ref.Wm)
    np.testing.assert_allclose(got.Wc, ref.Wc)
    np.testing.assert_allclose(got.sigma_points([2.0, -1.0], P), ref.sigma_points([2.0, -1.0], P))


def test_unscented_transform_matches_upstream():
    rng = np.random.default_rng(8)
    sigmas = rng.normal(size=(17, 8))
    wm = rng.random(17)
    wm /= wm.sum()
    wc = wm.copy()
    noise = np.eye(8) * 0.03
    gx, gp = unscented_transform(sigmas, wm, wc, noise)
    rx, rp = ref_ut(sigmas, wm, wc, noise)
    np.testing.assert_allclose(gx, rx, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(gp, rp, rtol=2e-13, atol=2e-13)


def test_unscented_transform_simd_tail_matches_upstream():
    rng = np.random.default_rng(18)
    sigmas = rng.normal(size=(15, 7))
    weights = rng.random(15)
    weights /= weights.sum()
    noise = np.eye(7) * 0.02
    gx, gp = unscented_transform(sigmas, weights, weights, noise)
    rx, rp = ref_ut(sigmas, weights, weights, noise)
    np.testing.assert_allclose(gx, rx, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(gp, rp, rtol=2e-13, atol=2e-13)


def test_unscented_transform_parallel_path_matches_upstream():
    rng = np.random.default_rng(28)
    sigmas = rng.normal(size=(193, 96))
    weights = rng.random(193)
    weights /= weights.sum()
    noise = np.eye(96) * 0.01
    gx, gp = unscented_transform(sigmas, weights, weights, noise)
    rx, rp = ref_ut(sigmas, weights, weights, noise)
    np.testing.assert_allclose(gx, rx, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(gp, rp, rtol=3e-13, atol=3e-13)


def test_unscented_transform_gpu_path_matches_upstream_or_falls_back():
    rng = np.random.default_rng(38)
    sigmas = rng.normal(size=(257, 128))
    weights = rng.random(257)
    weights /= weights.sum()
    noise = np.eye(128) * 0.01
    gx, gp = unscented_transform(sigmas, weights, weights, noise, device="gpu")
    rx, rp = ref_ut(sigmas, weights, weights, noise)
    np.testing.assert_allclose(gx, rx, rtol=3e-13, atol=3e-13)
    np.testing.assert_allclose(gp, rp, rtol=3e-13, atol=3e-13)


def test_unscented_transform_custom_functions_match():
    sigmas = np.array([[0.1, 1.0], [6.2, 2.0], [0.2, 3.0]])
    wm = np.array([0.3, 0.3, 0.4])

    def mean_fn(s, w):
        return np.array([np.arctan2(np.dot(w, np.sin(s[:, 0])), np.dot(w, np.cos(s[:, 0]))), np.dot(w, s[:, 1])])

    def residual(a, b):
        d = np.asarray(a) - np.asarray(b)
        d[0] = (d[0] + np.pi) % (2 * np.pi) - np.pi
        return d

    got = unscented_transform(sigmas, wm, wm, mean_fn=mean_fn, residual_fn=residual)
    ref = ref_ut(sigmas, wm, wm, mean_fn=mean_fn, residual_fn=residual)
    np.testing.assert_allclose(got[0], ref[0])
    np.testing.assert_allclose(got[1], ref[1])


@pytest.mark.parametrize(
    ("sigmas", "wm", "wc", "noise"),
    [
        (np.ones(3), np.ones(3), np.ones(3), None),
        (np.ones((3, 2)), np.ones(2), np.ones(3), None),
        (np.ones((3, 2)), np.ones(3), np.ones(3), np.eye(3)),
        (np.empty((0, 2)), np.empty(0), np.empty(0), None),
    ],
)
def test_unscented_transform_rejects_unsafe_shapes(sigmas, wm, wc, noise):
    with pytest.raises(ValueError):
        unscented_transform(sigmas, wm, wc, noise)


def fx(x, dt):
    return np.array([x[0] + dt * x[1], x[1]])


def hx(x):
    return np.array([x[0] ** 2])


def ukf_pair():
    gp = MerweScaledSigmaPoints(2, 0.2, 2.0, 0.0)
    rp = RefMerwe(2, 0.2, 2.0, 0.0)
    got = UnscentedKalmanFilter(2, 1, 0.1, hx, fx, gp)
    ref = RefUKF(2, 1, 0.1, hx, fx, rp)
    for obj in (got, ref):
        obj.x = np.array([0.5, 1.0])
        obj.P = np.array([[0.3, 0.02], [0.02, 0.2]])
        obj.Q = np.eye(2) * 0.01
        obj.R = np.eye(1) * 0.05
    return got, ref


def assert_ukf_close(got, ref):
    for name in ("x", "P", "K", "y", "S", "SI", "x_prior", "P_prior"):
        np.testing.assert_allclose(getattr(got, name), getattr(ref, name), rtol=2e-11, atol=2e-11)


def test_ukf_nonlinear_sequence_matches_upstream():
    got, ref = ukf_pair()
    for z in (0.4, 0.7, 1.0, 1.4):
        got.predict()
        ref.predict()
        got.update(z)
        ref.update(z)
        assert_ukf_close(got, ref)


def test_ukf_missing_measurement_matches():
    got, ref = ukf_pair()
    got.predict()
    ref.predict()
    got.update(None)
    ref.update(None)
    np.testing.assert_allclose(got.x_post, ref.x_post)
    np.testing.assert_allclose(got.P_post, ref.P_post)


def test_ukf_custom_inverse_matches_upstream():
    got, ref = ukf_pair()
    got.inv = ref.inv = np.linalg.pinv
    got.predict()
    ref.predict()
    got.update(0.8)
    ref.update(0.8)
    assert_ukf_close(got, ref)


def test_ukf_batch_filter_matches():
    got, ref = ukf_pair()
    zs = np.linspace(0.2, 1.2, 8)
    for a, b in zip(got.batch_filter(zs), ref.batch_filter(zs)):
        np.testing.assert_allclose(a, b, rtol=3e-10, atol=3e-10)


def test_ukf_rts_smoother_matches():
    got, ref = ukf_pair()
    zs = np.linspace(0.2, 1.2, 7)
    gm, gp = got.batch_filter(zs)
    rm, rp = ref.batch_filter(zs)
    for a, b in zip(got.rts_smoother(gm, gp), ref.rts_smoother(rm, rp)):
        np.testing.assert_allclose(a, b, rtol=2e-8, atol=2e-8)
