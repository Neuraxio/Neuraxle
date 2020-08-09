import math
import os
from typing import Counter

import joblib
import pytest
from scipy.stats import norm

from neuraxle.hyperparams.scipy_distributions import Gaussian, Histogram, Poisson, RandInt, Uniform, LogUniform, Normal, \
    LogNormal, Choice, get_index_in_list_with_bool
import numpy as np

NUM_TRIALS = 100


def get_many_samples_for(hd):
    return [hd.rvs() for _ in range(NUM_TRIALS)]


def test_wrapped_sk_learn_distributions_should_be_able_to_use_sklearn_methods():
    wrapped_sklearn_distribution = Gaussian(min_included=0, max_included=10, null_default_value=0)

    assert wrapped_sklearn_distribution.logpdf(5) == -13.418938533204672
    assert wrapped_sklearn_distribution.logcdf(5) == -0.6931477538632531
    assert wrapped_sklearn_distribution.sf(5) == 0.5000002866515718
    assert wrapped_sklearn_distribution.logsf(5) == -0.693146607256966
    assert np.all(wrapped_sklearn_distribution.ppf([0.0, 0.01, 0.05, 0.1, 1 - 0.10, 1 - 0.05, 1 - 0.01, 1.0], 10))
    assert wrapped_sklearn_distribution.isf(q=0.5) == 8.590676159074153
    assert wrapped_sklearn_distribution.moment(2) == 50.50000000091251
    assert wrapped_sklearn_distribution.stats()[0]
    assert wrapped_sklearn_distribution.stats()[1]
    assert np.array_equal(wrapped_sklearn_distribution.entropy(), np.array(0.7094692666023363))
    assert wrapped_sklearn_distribution.median()
    assert wrapped_sklearn_distribution.mean() == 5.398942280397029
    assert np.isclose(wrapped_sklearn_distribution.std(), 4.620759921685375)
    assert np.isclose(wrapped_sklearn_distribution.var(), 21.351422253853833)
    assert wrapped_sklearn_distribution.expect() == 0.39894228040143276
    interval = wrapped_sklearn_distribution.interval(alpha=[0.25, 0.50])
    assert np.all(interval[0])
    assert np.all(interval[1])
    assert wrapped_sklearn_distribution.support() == (0, 10)


def test_histogram():
    data = norm.rvs(size=10000, loc=0, scale=1.5, random_state=123)
    hist_dist = Histogram(
        histogram=np.histogram(data, bins=100),
        null_default_value=0.0
    )

    assert min(data) <= hist_dist.rvs() <= max(data)
    assert 1.0 > hist_dist.pdf(x=1.0) > 0.0
    assert hist_dist.pdf(x=np.max(data)) == 0.0
    assert hist_dist.pdf(x=np.min(data)) < 0.001
    assert hist_dist.cdf(x=np.max(data)) == 1.0
    assert 0.55 > hist_dist.cdf(x=np.median(data)) > 0.45
    assert hist_dist.cdf(x=np.min(data)) == 0.0


def test_continuous_gaussian():
    gaussian_distribution = Gaussian(
        min_included=0,
        max_included=10,
        null_default_value=0.0
    )

    assert 0.0 <= gaussian_distribution.rvs() <= 10.0
    assert gaussian_distribution.pdf(10) < 0.001
    assert gaussian_distribution.pdf(0) < 0.42
    assert 0.55 > gaussian_distribution.cdf(5.0) > 0.45
    assert gaussian_distribution.cdf(0) == 0.0


def test_discrete_poison():
    poisson_distribution = Poisson(
        min_included=0.0,
        max_included=10.0,
        null_default_value=0.0,
        mu=5.0
    )

    rvs = [poisson_distribution.rvs() for i in range(10)]
    assert not all(x == rvs[0] for x in rvs)
    assert 0.0 <= poisson_distribution.rvs() <= 10.0
    assert poisson_distribution.pdf(10) == 0.01813278870782187
    assert np.isclose(poisson_distribution.pdf(0), 0.006737946999085467)
    assert poisson_distribution.cdf(5.0) == 0.6159606548330632
    assert poisson_distribution.cdf(0) == 0.006737946999085467


def test_randint():
    hd = RandInt(min_included=-10, max_included=10, null_default_value=0)

    samples = hd.rvs(size=100)

    samples_mean = np.abs(np.mean(samples))
    assert -5.0 < samples_mean < 5.0
    assert min(samples) >= -10.0
    assert max(samples) <= 10.0

    assert hd.pdf(-11) == 0.
    assert abs(hd.pdf(-10) - 1 / (10 + 10 + 1)) == 0.0023809523809523864
    assert abs(hd.pdf(0) - 1 / (10 + 10 + 1)) == 0.0023809523809523864
    assert hd.pdf(5) == 0.05
    assert abs(hd.pdf(10) - 1 / (10 + 10 + 1)) == 0.047619047619047616
    assert hd.pdf(11) == 0.

    assert hd.cdf(-10.1) == 0.
    assert abs(hd.cdf(-10) - 1 / (10 + 10 + 1)) == 0.0023809523809523864
    assert abs(hd.cdf(0) - (0 + 10 + 1) / (10 + 10 + 1)) == 0.02619047619047621
    assert abs(hd.cdf(5) - (5 + 10 + 1) / (10 + 10 + 1)) == 0.03809523809523818
    assert abs(hd.cdf(10) - 1.) < 1e-6
    assert hd.cdf(10.1) == 1.

    assert hd.min() == -10
    assert hd.mean() == 0
    assert hd.median() == 0
    assert hd.std() > 2
    assert hd.max() == 10


def test_uniform(tmpdir):
    hd = Uniform(min_included=-10, max_included=10)
    _test_uniform(hd)

    hd = Uniform(min_included=-10, max_included=10)
    _test_uniform(hd)

    joblib.dump(hd, os.path.join(str(tmpdir), 'uniform.joblib'))
    hd = joblib.load(os.path.join(str(tmpdir), 'uniform.joblib'))

    _test_uniform(hd)


def _test_uniform(hd):
    samples = hd.rvs(size=100)
    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 4.0
    assert min(samples) >= -10.0
    assert max(samples) <= 10.0
    assert hd.pdf(-10.1) == 0.
    assert abs(hd.pdf(0) - 1 / (10 + 10)) < 1e-6
    assert hd.pdf(10.1) == 0.
    assert hd.cdf(-10.1) == 0.
    assert abs(hd.cdf(0) - (0 + 10) / (10 + 10)) < 1e-6
    assert hd.cdf(10.1) == 1.


def test_loguniform():
    hd = LogUniform(min_included=0.001, max_included=10)

    samples = hd.rvs(size=100)

    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 1.15  # if it was just uniform, this assert would break.
    assert min(samples) >= 0.001
    assert max(samples) <= 10.0
    assert hd.pdf(0.0001) == 0.
    assert abs(hd.pdf(2) - 0.054286810237906484) < 1e-6
    assert hd.pdf(10.1) == 0.
    assert hd.cdf(0.0001) == 0.
    assert abs(hd.cdf(2) - (math.log2(2) - math.log2(0.001)) / (math.log2(10) - math.log2(0.001))) < 1e-6
    assert hd.cdf(10.1) == 1.


def test_normal():
    hd = Normal(
        hard_clip_min=0.0,
        hard_clip_max=1.0,
        mean=0.5,
        std=0.2,
        null_default_value=0.0
    )

    samples = hd.rvs(size=100)

    samples_mean = np.abs(np.mean(samples))
    assert 0.6 > samples_mean > 0.4
    samples_std = np.std(samples)
    assert 0.1 < samples_std < 0.6
    assert abs(hd.pdf(-1.) - 0.24) == 0.24
    assert abs(hd.pdf(0.) - 0.40) == 0.31125636093539194
    assert abs(hd.pdf(1.)) == 0.08874363906460808
    assert abs(hd.cdf(-1.) - 0.15) == 0.15
    assert abs(hd.cdf(0.) - 0.5) == 0.5
    assert abs(hd.cdf(1.) - 0.85) == 0.15000000000000002


def test_lognormal():
    hd = LogNormal(
        hard_clip_min=-5,
        hard_clip_max=5,
        log2_space_mean=0.0,
        log2_space_std=2.0,
        null_default_value=-1.0
    )

    samples = hd.rvs(size=100)

    samples_median = np.median(samples)
    assert -5 < samples_median < 5
    samples_std = np.std(samples)
    assert 0 < samples_std < 4
    assert hd.pdf(0.) == 0.
    assert abs(hd.pdf(1.) - 0.28777602476804065) < 1e-6
    assert abs(hd.pdf(5.) - 0.029336304593386688) < 1e-6
    assert hd.cdf(0.) == 0.
    assert hd.cdf(1.) == 0.49999999998280026
    assert abs(hd.cdf(5.) - 0.8771717397015799) == 0.12282826029842009

@pytest.mark.parametrize("ctor", [Choice])
def test_choice_and_priority_choice(ctor):
    choice_list = [0, 1, False, "Test"]
    hd = ctor(choice_list)

    samples = hd.rvs(size=1000)
    z0 = Counter(samples).get(0)
    z1 = Counter(samples).get(1)
    zNone = Counter(samples).get(False)
    zTest = Counter(samples).get("Test")

    # You'd need to win the lotto for this test to fail. Or a broken random sampler. Or a bug.
    assert z0 > NUM_TRIALS * 0.2
    assert z1 > NUM_TRIALS * 0.2
    assert zNone > NUM_TRIALS * 0.2
    assert zTest > NUM_TRIALS * 0.2

    assert (hd.pdf(0) - 1 / 4) < 1e-6
    assert (hd.pdf(1) - 1 / 4) < 1e-6
    assert (hd.pdf(False) - 1 / 4) < 1e-6
    assert (hd.pdf("Test") - 1 / 4) < 1e-6
    assert abs(hd.pdf(0) - 1 / 4) < 1e-6
    assert abs(hd.pdf(1) - 1 / 4) < 1e-6
    assert abs(hd.pdf(False) - 1 / 4) < 1e-6
    assert abs(hd.pdf("Test") - 1 / 4) < 1e-6

    assert abs(hd.cdf(0) - 1 / 4) < 1e-6
    assert abs(hd.cdf(1) - 2 / 4) < 1e-6
    assert abs(hd.cdf(False) - 3 / 4) < 1e-6
    assert hd.cdf("Test") == 1.

    with pytest.raises(ValueError):
        assert hd.pdf(3) == 0.
        assert hd.cdf(3) == 0.

    assert hd.min() == 0
    assert hd.max() == len(choice_list)
    assert abs(hd.mean() - (len(choice_list) - 1) / 2) < 1e-6
    assert abs(hd.var() - (len(choice_list) ** 2 - 1) / 12) < 1e-6
    assert abs(hd.std() - math.sqrt((len(choice_list) ** 2 - 1) / 12)) < 1e-6
    # Convert samples in sample index
    samples_index = [get_index_in_list_with_bool(choice_list, sample) for sample in samples]
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs((hd.mean() - np.mean(samples_index)) / hd.mean()) < 1e-1
    assert abs((hd.var() - np.var(samples_index)) / hd.var()) < 1e-1
