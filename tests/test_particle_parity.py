import numpy as np
import pytest

from filterpy.monte_carlo import multinomial_resample as ref_multinomial
from filterpy.monte_carlo import residual_resample as ref_residual
from filterpy.monte_carlo import stratified_resample as ref_stratified
from filterpy.monte_carlo import systematic_resample as ref_systematic

from mojofilterpy.monte_carlo import (
    multinomial_resample,
    neff,
    residual_resample,
    stratified_resample,
    systematic_resample,
)

WEIGHTS = np.array([0.05, 0.1, 0.15, 0.2, 0.5])


@pytest.mark.parametrize(
    ("got_fn", "ref_fn"),
    [
        (systematic_resample, ref_systematic),
        (stratified_resample, ref_stratified),
        (multinomial_resample, ref_multinomial),
        (residual_resample, ref_residual),
    ],
)
def test_seeded_resampling_matches_upstream(got_fn, ref_fn):
    np.random.seed(31)
    expected = ref_fn(WEIGHTS)
    np.random.seed(31)
    got = got_fn(WEIGHTS)
    np.testing.assert_array_equal(got, expected)
    assert got.dtype == expected.dtype


def test_uniform_systematic_resampling_selects_every_particle_once():
    np.random.seed(9)
    got = systematic_resample(np.full(1000, 0.001))
    np.testing.assert_array_equal(got, np.arange(1000))


def test_multinomial_parallel_threshold_matches_upstream():
    rng = np.random.default_rng(19)
    weights = rng.random(32_768)
    weights /= weights.sum()
    np.random.seed(23)
    expected = ref_multinomial(weights)
    np.random.seed(23)
    got = multinomial_resample(weights)
    np.testing.assert_array_equal(got, expected)


def test_effective_sample_size_matches_formula():
    assert neff(WEIGHTS) == pytest.approx(1.0 / np.square(WEIGHTS).sum())


def test_resamplers_return_valid_indices():
    rng = np.random.default_rng(4)
    weights = rng.random(1000)
    weights /= weights.sum()
    for fn in (systematic_resample, stratified_resample, multinomial_resample, residual_resample):
        indexes = fn(weights)
        assert indexes.shape == (1000,)
        assert indexes.min() >= 0 and indexes.max() < 1000


@pytest.mark.parametrize("weights", [[], [[0.5, 0.5]], [1 + 1j]])
def test_resamplers_reject_unsafe_inputs(weights):
    with pytest.raises((TypeError, ValueError)):
        systematic_resample(weights)
