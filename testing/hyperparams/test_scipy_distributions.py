import math

from scipy.stats import norm

from neuraxle.hyperparams.scipy_distributions import Gaussian, Histogram, Poisson, RandInt, Uniform, LogUniform, Normal, \
    LogNormal
import numpy as np

NUM_TRIALS = 50000


def get_many_samples_for(hd):
    return [hd.rvs() for _ in range(NUM_TRIALS)]


def test_wrapped_sk_learn_distributions_should_be_able_to_use_sklearn_methods():
    wrapped_sklearn_distribution = Gaussian(min_included=0, max_included=10, null_default_value=0)

    assert wrapped_sklearn_distribution.logpdf(5) == -13.418938533204672
    assert wrapped_sklearn_distribution.logcdf(5) == -0.6931477538632531
    assert wrapped_sklearn_distribution.sf(5) == 0.5000002866515718
    assert wrapped_sklearn_distribution.logsf(5) == -0.693146607256966
    assert np.all(wrapped_sklearn_distribution.ppf([0.0, 0.01, 0.05, 0.1, 1 - 0.10, 1 - 0.05, 1 - 0.01, 1.0], 10))
    assert wrapped_sklearn_distribution.isf(q=0.5) == 8.798228093189323
    assert wrapped_sklearn_distribution.moment(2) == 50.50000000091249
    assert wrapped_sklearn_distribution.stats()[0]
    assert wrapped_sklearn_distribution.stats()[1]
    assert np.array_equal(wrapped_sklearn_distribution.entropy(), np.array(0.7094692666023363))
    assert wrapped_sklearn_distribution.median()
    assert wrapped_sklearn_distribution.mean() == 5.398942280397029
    assert wrapped_sklearn_distribution.std() == 4.620759921685374
    assert wrapped_sklearn_distribution.var() == 21.35142225385382
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
    hd = RandInt(-10, 10)

    samples = get_many_samples_for(hd)

    for s in samples:
        assert type(s) == int
    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 1.0
    assert min(samples) >= -10.0
    assert max(samples) <= 10.0

    assert hd.pdf(-11) == 0.
    assert abs(hd.pdf(-10) - 1 / (10 + 10 + 1)) < 1e-6
    assert abs(hd.pdf(0) - 1 / (10 + 10 + 1)) < 1e-6
    assert hd.pdf(0.5) == 0.
    assert abs(hd.pdf(10) - 1 / (10 + 10 + 1)) < 1e-6
    assert hd.pdf(11) == 0.

    assert hd.cdf(-10.1) == 0.
    assert abs(hd.cdf(-10) - 1 / (10 + 10 + 1)) < 1e-6
    assert abs(hd.cdf(0) - (0 + 10 + 1) / (10 + 10 + 1)) < 1e-6
    assert abs(hd.cdf(5) - (5 + 10 + 1) / (10 + 10 + 1)) < 1e-6
    assert abs(hd.cdf(10) - 1.) < 1e-6
    assert hd.cdf(10.1) == 1.


def test_uniform():
    hd = Uniform(-10, 10)

    samples = get_many_samples_for(hd)

    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 1.0
    assert min(samples) >= -10.0
    assert max(samples) <= 10.0
    assert hd.pdf(-10.1) == 0.
    assert abs(hd.pdf(0) - 1 / (10 + 10)) < 1e-6
    assert hd.pdf(10.1) == 0.
    assert hd.cdf(-10.1) == 0.
    assert abs(hd.cdf(0) - (0 + 10) / (10 + 10)) < 1e-6
    assert hd.cdf(10.1) == 1.


def test_loguniform():
    hd = LogUniform(0.001, 10)

    samples = get_many_samples_for(hd)

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
    hd = Normal(0.0, 1.0)

    samples = get_many_samples_for(hd)

    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 0.1
    samples_std = np.std(samples)
    assert 0.9 < samples_std < 1.1
    assert abs(hd.pdf(-1.) - 0.24197072451914337) < 1e-6
    assert abs(hd.pdf(0.) - 0.3989422804014327) < 1e-6
    assert abs(hd.pdf(1.) - 0.24197072451914337) < 1e-6
    assert abs(hd.cdf(-1.) - 0.15865525393145707) < 1e-6
    assert abs(hd.cdf(0.) - 0.5) < 1e-6
    assert abs(hd.cdf(1.) - 0.8413447460685429) < 1e-6


def test_lognormal():
    hd = LogNormal(0.0, 2.0)

    samples = get_many_samples_for(hd)

    samples_median = np.median(samples)
    assert 0.9 < samples_median < 1.1
    samples_std = np.std(samples)
    assert 5 < samples_std < 8
    assert hd.pdf(0.) == 0.
    assert abs(hd.pdf(1.) - 0.28777602476804065) < 1e-6
    assert abs(hd.pdf(5.) - 0.029336304593386688) < 1e-6
    assert hd.cdf(0.) == 0.
    assert hd.cdf(1.) == 0.5
    assert abs(hd.cdf(5.) - 0.8771717397015799) < 1e-6
