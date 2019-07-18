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


@pytest.mark.parametrize("ctor", [Choice, PriorityChoice])
def test_choice_and_priority_choice(ctor):
    hd = ctor([0, 1, False, "Test"])

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


def test_quantized_uniform():
    hd = Quantized(Uniform(-10, 10))

    samples = get_many_samples_for(hd)

    for s in samples:
        assert type(s) == int
    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 1.0
    assert min(samples) >= -10.0
    assert max(samples) <= 10.0


def test_randint():
    hd = RandInt(-10, 10)

    samples = get_many_samples_for(hd)

    for s in samples:
        assert type(s) == int
    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 1.0
    assert min(samples) >= -10.0
    assert max(samples) <= 10.0


def test_uniform():
    hd = Uniform(-10, 10)

    samples = get_many_samples_for(hd)

    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 1.0
    assert min(samples) >= -10.0
    assert max(samples) <= 10.0


def test_loguniform():
    hd = LogUniform(0.001, 10)

    samples = get_many_samples_for(hd)

    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 1.15  # if it was just uniform, this assert would break.
    assert min(samples) >= 0.001
    assert max(samples) <= 10.0


def test_normal():
    hd = Normal(0.0, 1.0)

    samples = get_many_samples_for(hd)

    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 0.1
    samples_std = np.std(samples)
    assert 0.9 < samples_std < 1.1


def test_lognormal():
    hd = LogNormal(0.0, 2.0)

    samples = get_many_samples_for(hd)

    samples_median = np.median(samples)
    assert 0.9 < samples_median < 1.1
    samples_std = np.std(samples)
    assert 5 < samples_std < 8


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
