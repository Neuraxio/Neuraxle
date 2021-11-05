import math
import os
from collections import Counter

import joblib
import numpy as np
import pytest
from neuraxle.base import Identity
from neuraxle.hyperparams.distributions import (Choice, LogNormal, LogUniform,
                                                Normal, PriorityChoice,
                                                RandInt, Uniform,
                                                get_index_in_list_with_bool)
from neuraxle.hyperparams.scipy_distributions import (
    Gaussian, Histogram, Poisson, ScipyContinuousDistributionWrapper,
    ScipyDiscreteDistributionWrapper, ScipyLogUniform, StdMeanLogNormal)
from neuraxle.hyperparams.space import HyperparameterSpace
from scipy.stats import gamma, norm, randint, uniform

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
    assert 8 < wrapped_sklearn_distribution.isf(q=0.5) > 8
    assert wrapped_sklearn_distribution.moment(2) > 50
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


HIST_DATA = norm.rvs(size=1000, loc=0, scale=1.5, random_state=123)


def test_histogram():
    hist_dist = Histogram(
        histogram=np.histogram(norm.rvs(size=1000, loc=0, scale=1.5, random_state=123), bins=10),
        null_default_value=0.0
    )

    _test_histogram(hist_dist)


def _test_histogram(hist_dist: Histogram):
    assert min(HIST_DATA) <= hist_dist.rvs() <= max(HIST_DATA)
    assert 1.0 > hist_dist.pdf(x=1.0) > 0.0
    assert hist_dist.pdf(x=np.max(HIST_DATA)) == 0.0
    assert hist_dist.pdf(x=np.min(HIST_DATA)) < 0.05
    assert hist_dist.cdf(x=np.max(HIST_DATA)) == 1.0
    assert 0.55 > hist_dist.cdf(x=np.median(HIST_DATA)) > 0.45
    assert hist_dist.cdf(x=np.min(HIST_DATA)) == 0.0


def test_continuous_gaussian():
    gaussian_distribution = Gaussian(
        min_included=0,
        max_included=10,
        null_default_value=0.0
    )

    _test_gaussian(gaussian_distribution)


def _test_gaussian(gaussian_distribution):
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

    _test_discrete_poisson(poisson_distribution)


def _test_discrete_poisson(poisson_distribution: Poisson):
    rvs = [poisson_distribution.rvs() for i in range(10)]
    assert not all(x == rvs[0] for x in rvs)
    assert 0.0 <= poisson_distribution.rvs() <= 10.0
    assert poisson_distribution.pdf(10) == 0.01813278870782187
    assert np.isclose(poisson_distribution.pdf(0), 0.006737946999085467)
    assert poisson_distribution.cdf(5.0) == 0.6159606548330632
    assert poisson_distribution.cdf(0) == 0.006737946999085467


def test_randint():
    hd = RandInt(min_included=-10, max_included=10, null_default_value=0)

    _test_randint(hd)


def _test_randint(hd):
    samples = hd.rvs_many(size=100)
    samples_mean = np.abs(np.mean(samples))
    invprob = 1 / (10 + 10 + 1)
    assert -5.0 < samples_mean < 5.0
    assert min(samples) >= -10.0
    assert max(samples) <= 10.0
    assert hd.pdf(-11) == 0.
    assert hd.pdf(-10) == invprob
    assert hd.pdf(0) == invprob
    assert hd.pdf(5) == invprob
    assert hd.pdf(10) == invprob
    assert hd.pdf(11) == 0.
    assert hd.cdf(-10.1) == 0.
    assert hd.cdf(-10) == invprob
    assert hd.cdf(5) == 16 * invprob
    assert abs(hd.cdf(10) - 1.) < 1e-6
    assert hd.cdf(10.1) == 1.
    assert hd.min() == -10
    assert hd.mean() == 0
    assert hd.std() > 2
    assert hd.max() == 10


def test_uniform():
    hd = Uniform(-10, 10)
    _test_uniform(hd)


def _test_uniform(hd):
    samples = hd.rvs_many(size=100)
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

    _test_loguniform(hd)


def _test_loguniform(hd):
    samples = hd.rvs_many(size=200)
    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 1.5  # if it was just uniform, this assert would break.
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

    _test_normal(hd)


def _test_normal(hd):
    samples = hd.rvs_many(size=100)
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
    hd = StdMeanLogNormal(
        hard_clip_min=-5,
        hard_clip_max=5,
        log2_space_mean=0.0,
        log2_space_std=2.0,
        null_default_value=-1.0
    )

    _test_lognormal(hd)


def _test_lognormal(hd):
    samples = hd.rvs_many(size=100)
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


@pytest.mark.parametrize("hd, test_method", [
    (RandInt(min_included=-10, max_included=10, null_default_value=0), _test_randint),
    (StdMeanLogNormal(hard_clip_min=-5, hard_clip_max=5, log2_space_mean=0.0,
     log2_space_std=2.0, null_default_value=-1.0), _test_lognormal),
    (Normal(hard_clip_min=0.0, hard_clip_max=1.0, mean=0.5, std=0.2, null_default_value=0.0), _test_normal),
    (ScipyLogUniform(min_included=0.001, max_included=10), _test_loguniform),
    (Uniform(min_included=-10, max_included=10), _test_uniform),
    (Poisson(min_included=0.0, max_included=10.0, null_default_value=0.0, mu=5.0), _test_discrete_poisson),
    (Gaussian(min_included=0, max_included=10, null_default_value=0.0), _test_gaussian),
    (Histogram(histogram=np.histogram(HIST_DATA, bins=10), null_default_value=0.0), _test_histogram)
])
def test_after_serialization(hd, test_method, tmpdir):
    joblib.dump(hd, os.path.join(str(tmpdir), '{}.joblib'.format(hd.__class__.__name__)))
    hd_loaded = joblib.load(os.path.join(str(tmpdir), '{}.joblib'.format(hd.__class__.__name__)))

    assert hd.__class__ == hd_loaded.__class__
    test_method(hd_loaded)


@pytest.mark.parametrize("hd", [
    Poisson(min_included=0.0, max_included=10.0, null_default_value=0.0, mu=5.0),
    Choice(choice_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    PriorityChoice(choice_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
])
def test_discrete_probabilities(hd):
    probas = hd.probabilities()

    sum_probas = sum(probas)
    assert sum_probas > 0.98
    assert len(probas) == 11


def test_can_set_scipy_distribution():
    space = HyperparameterSpace({
        'rand_int_scipy': randint(low=2, high=5),  # scipy
        'rand_int_neuraxle': RandInt(2, 5),  # neuraxle
        'gamma_scipy': gamma(0.2),  # scipy
    })

    p = Identity().set_hyperparams_space(space)

    rand_int_scipy = p.get_hyperparams_space()['rand_int_scipy']
    assert isinstance(rand_int_scipy, ScipyDiscreteDistributionWrapper)
    for _ in range(20):
        randint_sample = rand_int_scipy.rvs()
        assert randint_sample in rand_int_scipy

    gamma_scipy = p.get_hyperparams_space()['gamma_scipy']
    assert isinstance(gamma_scipy, ScipyContinuousDistributionWrapper)
    for _ in range(20):
        gamma_sample = gamma_scipy.rvs()
        assert isinstance(gamma_sample, float)
        assert gamma_sample in gamma_scipy


def test_can_update_scipy_distribution():
    p = Identity().set_hyperparams_space(HyperparameterSpace({
        'rand_int_neuraxle': RandInt(2, 5)  # neuraxle
    }))

    p.update_hyperparams_space(HyperparameterSpace({
        'rand_int_scipy': randint(low=2, high=5),  # scipy
        'gamma_scipy': gamma(0.2),  # scipy
    }))

    assert isinstance(p.get_hyperparams_space()['rand_int_scipy'], ScipyDiscreteDistributionWrapper)
    assert isinstance(p.get_hyperparams_space()['gamma_scipy'], ScipyContinuousDistributionWrapper)
    randint_sample = p.get_hyperparams_space()['rand_int_scipy'].rvs()
    gamma_sample = p.get_hyperparams_space()['gamma_scipy'].rvs()
    assert 5 >= randint_sample >= 2
    assert isinstance(gamma_sample, float)
