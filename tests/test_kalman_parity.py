import numpy as np
import pytest

from filterpy.kalman import ExtendedKalmanFilter as RefEKF
from filterpy.kalman import KalmanFilter as RefKF

from mojofilterpy.kalman import ExtendedKalmanFilter, KalmanFilter


def configured(cls, dim_x=4, dim_z=2, one_d=False):
    rng = np.random.default_rng(12)
    f = np.eye(dim_x) + rng.normal(scale=0.08, size=(dim_x, dim_x))
    h = rng.normal(size=(dim_z, dim_x))
    a = rng.normal(size=(dim_x, dim_x))
    p = a @ a.T + np.eye(dim_x)
    kf = cls(dim_x, dim_z, dim_u=1)
    kf.x = rng.normal(size=dim_x if one_d else (dim_x, 1))
    kf.P, kf.F, kf.H = p, f, h
    kf.Q, kf.R = np.eye(dim_x) * 0.03, np.eye(dim_z) * 0.2
    kf.B = np.linspace(0.1, 0.4, dim_x)
    if not one_d:
        kf.B = kf.B.reshape(-1, 1)
    return kf


def assert_filter_close(got, ref):
    for name in ("x", "P", "K", "y", "S", "SI", "x_prior", "P_prior"):
        np.testing.assert_allclose(getattr(got, name), getattr(ref, name), rtol=2e-12, atol=2e-12)


def test_kalman_defaults_match_upstream():
    got, ref = KalmanFilter(3, 2), RefKF(3, 2)
    for name in ("x", "P", "F", "H", "Q", "R", "K", "y", "S"):
        np.testing.assert_array_equal(getattr(got, name), getattr(ref, name))


@pytest.mark.parametrize("one_d", [False, True])
def test_kalman_predict_update_matches_upstream(one_d):
    got, ref = configured(KalmanFilter, one_d=one_d), configured(RefKF, one_d=one_d)
    u = 0.7 if one_d else np.array([[0.7]])
    got.predict(u=u)
    ref.predict(u=u)
    np.testing.assert_allclose(got.x, ref.x, rtol=2e-13, atol=2e-13)
    np.testing.assert_allclose(got.P, ref.P, rtol=2e-13, atol=2e-13)
    z = np.array([0.3, -1.2])
    got.update(z)
    ref.update(z)
    assert_filter_close(got, ref)


def test_kalman_scalar_noise_and_fading_memory_match():
    got, ref = configured(KalmanFilter), configured(RefKF)
    got.alpha = ref.alpha = 1.04
    got.predict(Q=0.15)
    ref.predict(Q=0.15)
    got.update([0.2, -0.1], R=0.4)
    ref.update([0.2, -0.1], R=0.4)
    assert_filter_close(got, ref)


def test_kalman_steady_state_methods_match():
    got, ref = configured(KalmanFilter, one_d=True), configured(RefKF, one_d=True)
    calibration = configured(RefKF, one_d=True)
    calibration.update([0.2, -0.1])
    got.K = ref.K = calibration.K.copy()
    got.predict_steadystate(u=0.4)
    ref.predict_steadystate(u=0.4)
    got.update_steadystate([0.3, -0.2])
    ref.update_steadystate([0.3, -0.2])
    for name in ("x", "P", "K", "y", "x_prior", "P_prior", "x_post", "P_post"):
        np.testing.assert_allclose(getattr(got, name), getattr(ref, name))


def test_kalman_simd_tail_matches_upstream():
    got = configured(KalmanFilter, dim_x=5, dim_z=3, one_d=True)
    ref = configured(RefKF, dim_x=5, dim_z=3, one_d=True)
    got.predict(u=0.4)
    ref.predict(u=0.4)
    got.update([0.2, -0.1, 0.7])
    ref.update([0.2, -0.1, 0.7])
    assert_filter_close(got, ref)


def test_kalman_missing_measurement_matches():
    got, ref = configured(KalmanFilter), configured(RefKF)
    got.predict()
    ref.predict()
    got.update(None)
    ref.update(None)
    np.testing.assert_allclose(got.x_post, ref.x_post)
    np.testing.assert_allclose(got.P_post, ref.P_post)
    np.testing.assert_array_equal(got.y, ref.y)


def test_kalman_custom_inverse_matches_upstream():
    got, ref = configured(KalmanFilter, one_d=True), configured(RefKF, one_d=True)
    got.inv = ref.inv = np.linalg.pinv
    got.update([0.4, -0.8])
    ref.update([0.4, -0.8])
    assert_filter_close(got, ref)


def test_kalman_likelihood_properties_match():
    got, ref = configured(KalmanFilter, one_d=True), configured(RefKF, one_d=True)
    got.update([0.4, -0.8])
    ref.update([0.4, -0.8])
    assert got.log_likelihood == pytest.approx(ref.log_likelihood, rel=1e-12)
    assert got.likelihood == pytest.approx(ref.likelihood, rel=1e-12)
    assert got.mahalanobis == pytest.approx(ref.mahalanobis, rel=1e-12)


def test_kalman_batch_filter_matches():
    got, ref = configured(KalmanFilter), configured(RefKF)
    zs = [np.array([np.sin(i), np.cos(i)]) for i in np.linspace(0, 1, 8)]
    gout = got.batch_filter(zs)
    rout = ref.batch_filter(zs)
    for a, b in zip(gout, rout):
        np.testing.assert_allclose(a, b, rtol=3e-12, atol=3e-12)


def test_kalman_rts_smoother_matches():
    got, ref = configured(KalmanFilter, dim_x=3, dim_z=1, one_d=True), configured(
        RefKF, dim_x=3, dim_z=1, one_d=True
    )
    got.B = ref.B = None
    means_g, covs_g, _, _ = got.batch_filter(np.linspace(-1, 1, 10))
    means_r, covs_r, _, _ = ref.batch_filter(np.linspace(-1, 1, 10))
    for a, b in zip(got.rts_smoother(means_g, covs_g), ref.rts_smoother(means_r, covs_r)):
        np.testing.assert_allclose(a, b, rtol=5e-11, atol=5e-11)


def ekf_configured(cls):
    obj = cls(2, 1)
    obj.x = np.array([0.5, 1.0])
    obj.P = np.array([[0.3, 0.02], [0.02, 0.2]])
    obj.F = np.array([[1.0, 0.1], [0.0, 1.0]])
    obj.Q = np.eye(2) * 0.01
    obj.R = np.eye(1) * 0.05
    return obj


def hj(x, scale=1.0):
    return np.array([[2.0 * x[0] * scale, 0.0]])


def hx(x, offset=0.0):
    return np.array([x[0] ** 2 + offset])


def test_ekf_predict_update_matches_upstream():
    got, ref = ekf_configured(ExtendedKalmanFilter), ekf_configured(RefEKF)
    got.predict()
    ref.predict()
    got.update(0.4, hj, hx)
    ref.update(0.4, hj, hx)
    for name in ("x", "P", "K", "y", "S", "x_prior", "P_prior"):
        np.testing.assert_allclose(getattr(got, name), getattr(ref, name), rtol=2e-12, atol=2e-12)


def test_ekf_callback_args_and_custom_residual_match():
    got, ref = ekf_configured(ExtendedKalmanFilter), ekf_configured(RefEKF)

    def residual(a, b):
        return np.asarray(a) - np.asarray(b) + 0.02

    kwargs = dict(args=(0.7,), hx_args=(0.1,), residual=residual)
    got.update([0.8], hj, hx, **kwargs)
    ref.update([0.8], hj, hx, **kwargs)
    for name in ("x", "P", "K", "y", "S", "x_prior", "P_prior"):
        np.testing.assert_allclose(getattr(got, name), getattr(ref, name), rtol=2e-12, atol=2e-12)


def test_ekf_predict_update_combined_matches_final_state():
    got, ref = ekf_configured(ExtendedKalmanFilter), ekf_configured(RefEKF)
    got.predict_update([0.8], hj, hx)
    ref.predict_update([0.8], hj, hx)
    for name in ("x", "P", "K", "y", "S"):
        np.testing.assert_allclose(getattr(got, name), getattr(ref, name), rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize(
    "bad",
    [
        np.array([1.0 + 2.0j, 2.0]),
        np.array([1.0, 2.0], dtype=np.longdouble),
        np.array([2**53 + 1], dtype=np.int64),
    ],
)
def test_native_boundary_rejects_lossy_numeric_conversion(bad):
    kf = KalmanFilter(bad.size, 1)
    kf.x = bad
    with pytest.raises((TypeError, ValueError)):
        kf.predict()


def test_native_boundary_rejects_bad_matrix_shapes_before_call():
    kf = KalmanFilter(2, 1)
    kf.F = np.ones((2, 3))
    with pytest.raises(ValueError):
        kf.predict()
