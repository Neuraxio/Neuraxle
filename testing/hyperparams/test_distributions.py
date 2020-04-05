"""
Tests for Hyperparameters Distributions
========================================

..
    Copyright 2019, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""

from collections import Counter

import pytest

from neuraxle.hyperparams.distributions import *

NUM_TRIALS = 50000


def get_many_samples_for(hd):
    return [hd.rvs() for _ in range(NUM_TRIALS)]


def test_boolean_distribution():
    hd = Boolean()

    samples = get_many_samples_for(hd)
    falses = Counter(samples).get(False)
    trues = Counter(samples).get(True)

    # You'd need to win the lotto for this test to fail. Or a broken random sampler. Or a bug.
    assert trues > NUM_TRIALS * 0.4
    assert falses > NUM_TRIALS * 0.4
    assert hd.pdf(False) == 0.5
    assert hd.pdf(0.) == 0.5
    assert hd.pdf(True) == 0.5
    assert hd.pdf(1.) == 0.5
    assert hd.pdf(-0.1) == 0.
    assert hd.pdf(1.1) == 0.

    assert hd.cdf(False) == 0.5
    assert hd.cdf(0.) == 0.5
    assert hd.cdf(True) == 1.
    assert hd.cdf(1.) == 1.
    assert hd.cdf(-0.1) == 0.
    assert hd.cdf(1.1) == 1.

    assert hd.min() == 0
    assert hd.max() == 1
    assert abs(hd.mean() - 0.5) < 1e-6
    assert abs(hd.std() - 0.5) < 1e-6
    assert abs(hd.var() - 0.25) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-2
    assert abs(hd.var() - np.var(samples)) < 1e-2

@pytest.mark.parametrize("ctor", [Choice, PriorityChoice])
def test_choice_and_priority_choice(ctor):
    choice_list = [0, 1, False, "Test"]
    hd = ctor(choice_list)

    samples = get_many_samples_for(hd)
    z0 = Counter(samples).get(0)
    z1 = Counter(samples).get(1)
    zNone = Counter(samples).get(False)
    zTest = Counter(samples).get("Test")

    # You'd need to win the lotto for this test to fail. Or a broken random sampler. Or a bug.
    assert z0 > NUM_TRIALS * 0.2
    assert z1 > NUM_TRIALS * 0.2
    assert zNone > NUM_TRIALS * 0.2
    assert zTest > NUM_TRIALS * 0.2

    assert abs(hd.pdf(0) - 1 / 4) < 1e-6
    assert abs(hd.pdf(1) - 1 / 4) < 1e-6
    assert abs(hd.pdf(False) - 1/4) < 1e-6
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
    assert abs(hd.var() - (len(choice_list)**2 - 1) / 12) < 1e-6
    assert abs(hd.std() - math.sqrt((len(choice_list)**2 - 1) / 12)) < 1e-6
    # Convert samples in sample index
    samples_index = [get_index_in_list_with_bool(choice_list, sample) for sample in samples]
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples_index)) < 1e-2
    assert abs(hd.var() - np.var(samples_index)) < 1e-2

def test_quantized_uniform():
    low = -10
    high = 10
    hd = Quantized(Uniform(low, high))

    samples = get_many_samples_for(hd)

    for s in samples:
        assert type(s) == int
    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 1.0
    assert min(samples) >= -10.0
    assert max(samples) <= 10.0

    assert abs(hd.pdf(-10) - 1/40) < 1e-6
    assert abs(hd.pdf(-9) - 1 / 20) < 1e-6
    assert abs(hd.pdf(0) - 1 / 20) < 1e-6
    assert abs(hd.pdf(9) - 1 / 20) < 1e-6
    assert abs(hd.pdf(10) - 1 / 40) < 1e-6

    assert abs(hd.cdf(-10) - 1/40) < 1e-6
    assert abs(hd.cdf(-9) - 1.5 / 20) < 1e-6
    assert abs(hd.cdf(0) - 10.5 / 20) < 1e-6
    assert abs(hd.cdf(9) - 19.5 / 20) < 1e-6
    assert abs(hd.cdf(9.2) - 19.5 / 20) < 1e-6
    assert hd.cdf(10) == 1.

    assert hd.min() == low
    assert hd.max() == high
    assert abs(hd.mean() - 1.75) < 1e-6
    assert abs(hd.var() - 30.4375) < 1e-6
    assert abs(hd.std() - 5.5170191) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-2
    assert abs(hd.var() - np.var(samples)) < 1e-2


def test_randint():
    low = -10
    high = 10
    hd = RandInt(low, high)

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

    assert hd.min() == low
    assert hd.max() == high
    assert abs(hd.mean() - (10 - 10) / 2) < 1e-6
    assert abs(hd.var() - ((high - low + 1)**2 - 1) / 12) < 1e-6
    assert abs(hd.std() - math.sqrt(((high - low + 1)**2 - 1) / 12)) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 5e-2
    assert abs(hd.var() - np.var(samples)) < 5e-2

def test_uniform():
    low = -10
    high = 10
    hd = Uniform(low, high)

    samples = get_many_samples_for(hd)

    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 1.0
    assert min(samples) >= -10.0
    assert max(samples) <= 10.0
    assert hd.pdf(-10.1) == 0.
    assert abs(hd.pdf(0) - 1/(10 + 10)) < 1e-6
    assert hd.pdf(10.1) == 0.
    assert hd.cdf(-10.1) == 0.
    assert abs(hd.cdf(0) - (0 + 10) / (10 + 10)) < 1e-6
    assert hd.cdf(10.1) == 1.

    assert hd.min() == low
    assert hd.max() == high
    assert abs(hd.mean() - (10 - 10) / 2) < 1e-6
    assert abs(hd.var() - 1/12 * (high - low)**2) < 1e-6
    assert abs(hd.std() - math.sqrt(1/12 * (high - low)**2)) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 5e-2
    assert abs(hd.var() - np.var(samples)) < 5e-2

def test_loguniform():
    min_included = 0.001
    max_included = 10
    hd = LogUniform(min_included, max_included)

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

    assert hd.min() == min_included
    assert hd.max() == max_included
    assert abs(hd.mean() - (max_included - min_included) / (math.log(2) * (math.log2(max_included) - math.log2(min_included)))) < 1e-6
    esperance_squared = (max_included ** 2 - min_included ** 2) / (
                2 * math.log(2) * (math.log2(max_included) - math.log2(min_included)))
    assert abs(hd.var() - (esperance_squared - hd.mean()**2)) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 5e-2
    assert abs(hd.var() - np.var(samples)) < 1e-1

def test_normal():
    hd_mean = 0.0
    hd_std = 1.0
    hd = Normal(hd_mean, hd_std)

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

    assert hd.min() == -np.inf
    assert hd.max() == np.inf
    assert abs(hd.mean() - hd_mean) < 1e-6
    assert abs(hd.var() - hd_std**2) < 1e-6
    assert abs(hd.std() - hd_std) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 5e-2
    assert abs(hd.var() - np.var(samples)) < 5e-2

def test_lognormal():
    log2_space_mean = 0.0
    log2_space_std = 2.0
    hd = LogNormal(log2_space_mean, log2_space_std)

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

    assert hd.min() == 0
    assert hd.max() == np.inf
    # assert abs(hd.mean() - hd_mean) < 1e-6
    # assert abs(hd.var() - hd_std**2) < 1e-6
    # assert abs(hd.std() - hd_std) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 6e-2
    assert abs((hd.var() - np.var(samples)) / hd.var()) < 1e-2

@pytest.mark.parametrize("hd", [
    FixedHyperparameter(0),
    Boolean(),
    Choice([0, 1, False, "Test"]),
    PriorityChoice([0, 1, False, "Test"]),
    Quantized(Uniform(-10, 10)),
    RandInt(-10, 10),
    Uniform(-10, 10),
    LogUniform(0.001, 10),
    Normal(0.0, 1.0),
    LogNormal(0.0, 2.0)
])
def test_can_restore_each_distributions(hd):
    print(hd.__dict__)
    reduced = hd.narrow_space_from_best_guess(1, 0.5)
    reduced = reduced.narrow_space_from_best_guess(1, 0.5)

    assert reduced.unnarrow() == hd


def test_choice_threshold_narrowing():
    hd = Choice([0, 1, False, "Test"])

    hd = hd.narrow_space_from_best_guess(False, 1.0)
    assert isinstance(hd, Choice)
    assert len(hd) == 4

    hd = hd.narrow_space_from_best_guess(False, 0.5)
    assert isinstance(hd, Choice)
    assert len(hd) == 4

    hd = hd.narrow_space_from_best_guess(False, 0.5)
    assert isinstance(hd, FixedHyperparameter)

    hd = hd.narrow_space_from_best_guess(False, 0.5)
    assert isinstance(hd, FixedHyperparameter)
    assert hd.get_current_narrowing_value() == 0.5 ** 3

    hd = hd.unnarrow()
    assert isinstance(hd, Choice)
    assert len(hd) == 4
    assert hd.get_current_narrowing_value() == 1.0


def test_priority_choice_threshold_narrowing():
    hd = PriorityChoice([0, 1, False, "Test"])

    hd = hd.narrow_space_from_best_guess(False, 1.0)
    assert False == hd.choice_list[0]
    assert isinstance(hd, PriorityChoice)
    assert len(hd) == 4

    hd = hd.narrow_space_from_best_guess(False, 0.75)
    assert False == hd.choice_list[0]
    assert isinstance(hd, PriorityChoice)
    assert len(hd) == 3

    hd = hd.narrow_space_from_best_guess(False, 0.5)
    assert isinstance(hd, FixedHyperparameter)

    hd = hd.narrow_space_from_best_guess(False, 0.5)
    assert isinstance(hd, FixedHyperparameter)
    assert hd.get_current_narrowing_value() == 0.75 * 0.5 ** 2

    hd = hd.unnarrow()
    assert isinstance(hd, PriorityChoice)
    assert len(hd) == 4
    assert hd.get_current_narrowing_value() == 1.0

