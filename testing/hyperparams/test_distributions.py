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

import random
from collections import Counter

import numpy as np
import pytest
from neuraxle.hyperparams.distributions import *


NUM_TRIALS = 5000


def hd_rvs_many(hd: HyperparameterDistribution):
    random.seed(111)
    np.random.seed(111)
    return hd.rvs_many(NUM_TRIALS)


def test_boolean_distribution():
    hd = Boolean()

    samples = hd_rvs_many(hd)
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


def test_boolean_distribution_with_proba():
    proba_is_true = 0.7
    hd = Boolean(proba_is_true=proba_is_true)

    samples = hd_rvs_many(hd)
    falses = Counter(samples).get(False)
    trues = Counter(samples).get(True)

    # You'd need to win the lotto for this test to fail. Or a broken random sampler. Or a bug.
    assert trues > NUM_TRIALS * (proba_is_true - 0.1)
    assert falses > NUM_TRIALS * (1 - proba_is_true - 0.1)
    assert abs(hd.pdf(False) - (1 - proba_is_true)) < 1e-6
    assert abs(hd.pdf(0.) - (1 - proba_is_true)) < 1e-6
    assert abs(hd.pdf(True) - proba_is_true) < 1e-6
    assert abs(hd.pdf(1.) - proba_is_true) < 1e-6
    assert abs(hd.pdf(-0.1) - 0.) < 1e-6
    assert abs(hd.pdf(1.1) - 0.) < 1e-6

    assert abs(hd.cdf(False) - (1 - proba_is_true)) < 1e-6
    assert abs(hd.cdf(0.) - (1 - proba_is_true)) < 1e-6
    assert abs(hd.cdf(True) - 1.) < 1e-6
    assert abs(hd.cdf(1.) - 1.) < 1e-6
    assert abs(hd.cdf(-0.1) - 0.) < 1e-6
    assert abs(hd.cdf(1.1) - 1.) < 1e-6

    assert hd.min() == 0
    assert hd.max() == 1
    assert abs(hd.mean() - proba_is_true) < 1e-6
    assert abs(hd.std() - math.sqrt(proba_is_true * (1 - proba_is_true))) < 1e-6
    assert abs(hd.var() - proba_is_true * (1 - proba_is_true)) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-2
    assert abs(hd.var() - np.var(samples)) < 1e-2


@pytest.mark.parametrize("ctor", [Choice, PriorityChoice])
def test_choice_and_priority_choice(ctor):
    choice_list = [0, 1, False, "Test"]
    hd = ctor(choice_list)

    samples = hd_rvs_many(hd)
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


@pytest.mark.parametrize("ctor", [Choice, PriorityChoice])
def test_choice_and_priority_choice_with_probas(ctor):
    probas = [0.1, 0.4, 0.3, 0.2]
    probas_array = np.array(probas)
    choice_list = [0, 1, False, "Test"]
    hd = ctor(choice_list, probas=probas)

    samples = hd_rvs_many(hd)
    z0 = Counter(samples).get(0)
    z1 = Counter(samples).get(1)
    zNone = Counter(samples).get(False)
    zTest = Counter(samples).get("Test")

    # You'd need to win the lotto for this test to fail. Or a broken random sampler. Or a bug.
    assert z0 > NUM_TRIALS * (probas[0] - 0.05)
    assert z1 > NUM_TRIALS * (probas[1] - 0.05)
    assert zNone > NUM_TRIALS * (probas[2] - 0.05)
    assert zTest > NUM_TRIALS * (probas[3] - 0.05)

    assert abs(hd.pdf(0) - probas[0]) < 1e-6
    assert abs(hd.pdf(1) - probas[1]) < 1e-6
    assert abs(hd.pdf(False) - probas[2]) < 1e-6
    assert abs(hd.pdf("Test") - probas[3]) < 1e-6

    assert abs(hd.cdf(0) - probas_array[0]) < 1e-6
    assert abs(hd.cdf(1) - np.sum(probas_array[0:2])) < 1e-6
    assert abs(hd.cdf(False) - np.sum(probas_array[0:3])) < 1e-6
    assert abs(hd.cdf("Test") - 1.) < 1e-6

    with pytest.raises(ValueError):
        assert hd.pdf(3) == 0.
        assert hd.cdf(3) == 0.

    assert hd.min() == 0
    assert hd.max() == len(choice_list)
    assert abs(hd.mean() - 1.6) < 1e-6
    assert abs(hd.var() - 0.84) < 1e-6
    assert abs(hd.std() - 0.9165151389911679) < 1e-6
    # Convert samples in sample index
    samples_index = [get_index_in_list_with_bool(choice_list, sample) for sample in samples]
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs((hd.mean() - np.mean(samples_index)) / hd.mean()) < 1e-1
    assert abs((hd.var() - np.var(samples_index)) / hd.var()) < 1e-1


def test_quantized_uniform():
    low = -10
    high = 10
    hd = Quantized(Uniform(low, high))

    samples = hd_rvs_many(hd)

    for s in samples:
        assert type(s) == int
    samples_mean = np.abs(np.mean(samples))
    assert samples_mean < 1.0
    assert min(samples) >= -10.0
    assert max(samples) <= 10.0

    assert abs(hd.pdf(-10) - 1 / 40) < 1e-6
    assert abs(hd.pdf(-9) - 1 / 20) < 1e-6
    assert abs(hd.pdf(0) - 1 / 20) < 1e-6
    assert abs(hd.pdf(9) - 1 / 20) < 1e-6
    assert abs(hd.pdf(10) - 1 / 40) < 1e-6

    assert abs(hd.cdf(-10) - 1 / 40) < 1e-6
    assert abs(hd.cdf(-9) - 1.5 / 20) < 1e-6
    assert abs(hd.cdf(0) - 10.5 / 20) < 1e-6
    assert abs(hd.cdf(9) - 19.5 / 20) < 1e-6
    assert abs(hd.cdf(9.2) - 19.5 / 20) < 1e-6
    assert hd.cdf(10) == 1.

    assert hd.min() == low
    assert hd.max() == high
    assert abs(hd.mean() - 0.0) < 1e-6
    assert abs(hd.var() - 33.50000000000001) < 1e-6
    assert abs(hd.std() - 5.787918451395114) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-1
    assert abs((hd.var() - np.var(samples)) / hd.var()) < 1e-1


def test_randint():
    low = -10
    high = 10
    hd = RandInt(low, high)

    samples = hd_rvs_many(hd)

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
    assert abs(hd.var() - ((high - low + 1) ** 2 - 1) / 12) < 1e-6
    assert abs(hd.std() - math.sqrt(((high - low + 1) ** 2 - 1) / 12)) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-1
    assert abs((hd.var() - np.var(samples)) / hd.var()) < 1e-1


def test_uniform():
    low = -10
    high = 10
    hd = Uniform(low, high)

    samples = hd_rvs_many(hd)

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

    assert hd.min() == low
    assert hd.max() == high
    assert abs(hd.mean() - (10 - 10) / 2) < 1e-6
    assert abs(hd.var() - 1 / 12 * (high - low) ** 2) < 1e-6
    assert abs(hd.std() - math.sqrt(1 / 12 * (high - low) ** 2)) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-1
    assert abs((hd.var() - np.var(samples)) / hd.var()) < 1e-1


def test_loguniform():
    min_included = 0.001
    max_included = 10
    hd = LogUniform(min_included, max_included)

    samples = hd_rvs_many(hd)

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
    assert abs(hd.mean() - (max_included - min_included) / (
        math.log(2) * (math.log2(max_included) - math.log2(min_included)))) < 1e-6
    esperance_squared = (max_included ** 2 - min_included ** 2) / (
        2 * math.log(2) * (math.log2(max_included) - math.log2(min_included)))
    assert abs(hd.var() - (esperance_squared - hd.mean() ** 2)) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 5e-2
    assert abs(hd.var() - np.var(samples)) < 2.5e-1


def test_normal():
    hd_mean = 0.0
    hd_std = 1.0
    hd = Normal(hd_mean, hd_std)

    samples = hd_rvs_many(hd)

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
    assert abs(hd.var() - hd_std ** 2) < 1e-6
    assert abs(hd.std() - hd_std) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 5e-2
    assert abs(hd.var() - np.var(samples)) < 5e-2


def test_normal_truncated():
    hd_mean = 2.0
    hd_std = 1.0
    hard_clip_min = 1.8
    hard_clip_max = 2.5
    hd = Normal(hd_mean, hd_std, hard_clip_min=hard_clip_min, hard_clip_max=hard_clip_max)

    samples = hd_rvs_many(hd)

    samples_mean = np.abs(np.mean(samples))
    assert 2.0 < samples_mean < 2.2
    samples_std = np.std(samples)
    assert 0 < samples_std < 0.4
    assert abs(hd.pdf(1.7) - 0.) < 1e-6
    assert abs(hd.pdf(1.8) - 1.4444428136247596) < 1e-6
    assert abs(hd.pdf(2.) - 1.473622494051997) < 1e-6
    assert abs(hd.pdf(2.25) - 1.4282838963071145) < 1e-6
    assert abs(hd.pdf(2.5) - 1.3004672865798739) < 1e-6
    assert abs(hd.pdf(2.6) - 0.) < 1e-6
    assert abs(hd.cdf(1.7) - 0.) < 1e-6
    assert abs(hd.cdf(1.8) - 0.) < 1e-6
    assert abs(hd.cdf(2.) - 0.2927714018778846) < 1e-6
    assert abs(hd.cdf(2.25) - 0.65737517785574) < 1e-6
    assert abs(hd.cdf(2.5) - 1.) < 1e-6
    assert abs(hd.cdf(2.6) - 1.) < 1e-6

    assert np.all((np.array(samples) >= hard_clip_min) & (np.array(samples) <= hard_clip_max))
    assert hd.min() == hard_clip_min
    assert hd.max() == hard_clip_max
    assert abs(hd.mean() - 2.1439755270448857) < 1e-6
    assert abs(hd.var() - 0.04014884159725845) < 1e-6
    assert abs(hd.std() - 0.20037175848222336) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-2
    assert abs(hd.var() - np.var(samples)) < 1e-2


def test_normal_onside_lower_tail_truncated():
    hd_mean = 2.0
    hd_std = 1.0
    hard_clip_min = 1.8
    hard_clip_max = None
    hd = Normal(hd_mean, hd_std, hard_clip_min=hard_clip_min, hard_clip_max=hard_clip_max)

    samples = hd_rvs_many(hd)

    samples_mean = np.abs(np.mean(samples))
    assert 2.5 < samples_mean < 2.8
    samples_std = np.std(samples)
    assert 0.5 < samples_std < 0.7
    assert abs(hd.pdf(1.7) - 0.) < 1e-6
    assert abs(hd.pdf(1.8) - 0.6750731797902921) < 1e-6
    assert abs(hd.pdf(2.) - 0.688710562638179) < 1e-6
    assert abs(hd.pdf(2.5) - 0.607784938305487) < 1e-6
    assert abs(hd.pdf(5.) - 0.007650883256198442) < 1e-6
    assert abs(hd.cdf(1.7) - 0.) < 1e-6
    assert abs(hd.cdf(1.8) - 0.) < 1e-6
    assert abs(hd.cdf(2.) - 0.13682931532705794) < 1e-6
    assert abs(hd.cdf(2.5) - 0.46735888290117106) < 1e-6
    assert abs(hd.cdf(5.) - 0.9976696151835984) < 1e-6

    assert np.all(np.array(samples) >= hard_clip_min)
    assert hd.min() == hard_clip_min
    assert hd.max() == np.inf
    assert abs(hd.mean() - 2.6750731797902922) < 1e-6
    assert abs(hd.var() - 0.4092615659697655) < 1e-6
    assert abs(hd.std() - 0.6397355437755241) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-2
    assert abs(hd.var() - np.var(samples)) < 1e-2


def test_normal_onside_upper_tail_truncated():
    hd_mean = 2.0
    hd_std = 1.0
    hard_clip_min = None
    hard_clip_max = 2.5
    hd = Normal(hd_mean, hd_std, hard_clip_min=hard_clip_min, hard_clip_max=hard_clip_max)

    samples = hd_rvs_many(hd)

    samples_mean = np.abs(np.mean(samples))
    assert 1.4 < samples_mean < 1.6
    samples_std = np.std(samples)
    assert 0.6 < samples_std < 0.8
    assert abs(hd.pdf(-1.) - 0.006409383965359982) < 1e-6
    assert abs(hd.pdf(1.8) - 0.5655298962361072) < 1e-6
    assert abs(hd.pdf(2.) - 0.5769543579652687) < 1e-6
    assert abs(hd.pdf(2.5) - 0.5091604338370336) < 1e-6
    assert abs(hd.pdf(2.6) - 0.) < 1e-6
    assert abs(hd.cdf(-1.) - 0.0019522361765567414) < 1e-6
    assert abs(hd.cdf(1.8) - 0.6084788605670465) < 1e-6
    assert abs(hd.cdf(2.) - 0.723105053423659) < 1e-6
    assert abs(hd.cdf(2.5) - 1.) < 1e-6
    assert abs(hd.cdf(2.6) - 1.) < 1e-6

    assert np.all(np.array(samples) <= hard_clip_max)
    assert hd.min() == -np.inf
    assert hd.max() == hard_clip_max
    assert abs(hd.mean() - 1.4908395661629665) < 1e-6
    assert abs(hd.var() - 0.486175435696367) < 1e-6
    assert abs(hd.std() - 0.6972628168032245) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-2
    assert abs(hd.var() - np.var(samples)) < 1e-2


@pytest.mark.parametrize("seed", (15, 20, 32, 40, 50))
def test_lognormal(seed):
    np.random.seed(seed)

    log2_space_mean = 0.0
    log2_space_std = 2.0
    hd = LogNormal(log2_space_mean, log2_space_std)

    samples = hd_rvs_many(hd)

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
    assert abs(hd.mean() - 2.614063815405198) < 1e-6
    assert abs(hd.var() - 39.86106421503915) < 1e-6
    assert abs(hd.std() - 6.313561927710787) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-1
    assert abs((hd.var() - np.var(samples)) / hd.var()) < 2.5e-1


def test_lognormal_clipped():
    log2_space_mean = 10.0
    log2_space_std = 5.0
    hard_clip_min = 5
    hard_clip_max = 100
    hd = LogNormal(log2_space_mean, log2_space_std, hard_clip_min=hard_clip_min, hard_clip_max=hard_clip_max)

    samples = hd_rvs_many(hd)

    samples_median = np.median(samples)
    assert 25 < samples_median < 35
    samples_std = np.std(samples)
    assert 20 < samples_std < 30
    assert hd.pdf(0.) == 0.
    assert abs(hd.pdf(6.) - 0.03385080142719004) < 1e-6
    assert abs(hd.pdf(10.) - 0.024999599033243936) < 1e-6
    assert hd.cdf(0.) == 0.
    assert abs(hd.cdf(6.) - 0.03560663481768936) < 1e-6
    assert abs(hd.cdf(10.) - 0.15112888563249155) < 1e-6

    assert hd.min() == hard_clip_min
    assert hd.max() == hard_clip_max
    assert abs(hd.mean() - 38.110960930274594) < 1e-6
    assert abs(hd.var() - 728.7599668053633) < 1e-6
    assert abs(hd.std() - 26.995554574880718) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs((hd.mean() - np.mean(samples)) / hd.mean()) < 1e-1
    assert abs((hd.var() - np.var(samples)) / hd.var()) < 1e-1


def test_gaussian_distribution_mixture():
    distribution_amplitudes = [1, 1, 1]
    means = [-2, 0, 2]
    stds = [1, 1, 1]
    distribution_mins = [None for _ in range(len(means))]
    distribution_max = [None for _ in range(len(means))]

    hd = DistributionMixture.build_gaussian_mixture(distribution_amplitudes, means, stds, distribution_mins,
                                                    distribution_max)

    samples = hd_rvs_many(hd)

    samples_median = np.median(samples)
    assert -0.5 < samples_median < 0.5
    samples_std = np.std(samples)
    assert 1 < samples_std < 4
    assert abs(hd.pdf(-2.) - 0.1510223590467952) < 1e-6
    assert abs(hd.pdf(0.) - 0.16897473780926958) < 1e-6
    assert abs(hd.pdf(2.) - 0.1510223590467952) < 1e-6
    assert abs(hd.cdf(-2.) - 0.17426060106333743) < 1e-6
    assert abs(hd.cdf(0.) - 0.5) < 1e-6
    assert abs(hd.cdf(2.) - 0.8257393989366625) < 1e-6

    assert hd.min() == -np.inf
    assert hd.max() == np.inf
    assert abs(hd.mean() - 0.0) < 1e-6
    assert abs(hd.var() - 11. / 3) < 1e-6
    assert abs(hd.std() - math.sqrt(11. / 3)) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-1
    assert abs(hd.var() - np.var(samples)) < 1e-1


def test_gaussian_distribution_mixture_truncated():
    distribution_amplitudes = [1, 1, 1]
    means = [-2, 0, 2]
    stds = [1, 1, 1]
    distribution_mins = [-2.5, -0.5, 1.5]
    distribution_max = [-1.5, 0.5, 2.5]

    hd = DistributionMixture.build_gaussian_mixture(distribution_amplitudes, means, stds, distribution_mins,
                                                    distribution_max)

    samples = hd_rvs_many(hd)

    assert np.all(np.logical_and(np.array(samples) >= distribution_mins[0], np.array(samples) <= distribution_max[0]) |
                  np.logical_and(np.array(samples) >= distribution_mins[1], np.array(samples) <= distribution_max[1]) |
                  np.logical_and(np.array(samples) >= distribution_mins[2], np.array(samples) <= distribution_max[2]))
    samples_median = np.median(samples)
    assert -0.5 < samples_median < 0.5
    samples_std = np.std(samples)
    assert 1 < samples_std < 4
    assert abs(hd.pdf(-2.) - 0.3472763257323177) < 1e-6
    assert abs(hd.pdf(0.) - 0.3472763257323177) < 1e-6
    assert abs(hd.pdf(2.) - 0.3472763257323177) < 1e-6
    assert abs(hd.cdf(-2.) - 0.16666666666666666) < 1e-6
    assert abs(hd.cdf(0.) - 0.5) < 1e-6
    assert abs(hd.cdf(2.) - 0.8333333333333333) < 1e-6

    assert hd.min() == min(distribution_mins)
    assert hd.max() == max(distribution_max)
    assert abs(hd.mean() - 0.0) < 1e-6
    assert abs(hd.var() - 2.747255821267478) < 1e-6
    assert abs(hd.std() - 1.6574847876428545) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-1
    assert abs(hd.var() - np.var(samples)) < 1e-1


def test_gaussian_distribution_mixture_log():
    distribution_amplitudes = [1, 1, 1]
    means = [-2, 0, 2]
    stds = [1, 1, 1]
    distribution_mins = [None for _ in range(len(means))]
    distribution_max = [None for _ in range(len(means))]

    hd = DistributionMixture.build_gaussian_mixture(distribution_amplitudes, means, stds, distribution_mins,
                                                    distribution_max, use_logs=True)

    samples = hd_rvs_many(hd)

    samples_median = np.median(samples)
    assert 0.5 < samples_median < 1.5
    samples_std = np.std(samples)
    assert 1 < samples_std < 4
    assert abs(hd.pdf(-2.) - 0.) < 1e-6
    assert abs(hd.pdf(1.) - 0.24377901627294607) < 1e-6
    assert abs(hd.pdf(5.) - 0.03902571107126729) < 1e-6
    assert abs(hd.cdf(-2.) - 0.) < 1e-6
    assert abs(hd.cdf(1.) - 0.5) < 1e-6
    assert abs(hd.cdf(5.) - 0.8720400927468334) < 1e-6

    assert hd.min() == 0
    assert hd.max() == np.inf
    assert abs(hd.mean() - 2.225189976999746) < 1e-6
    assert abs(hd.var() - 9.916017516376925) < 1
    assert abs(hd.std() - 3.1489708662318434) < 0.3
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 2e-1
    assert abs(hd.var() - np.var(samples)) < 1


def test_gaussian_distribution_mixture_quantized():
    distribution_amplitudes = [1, 1, 1]
    means = [-2, 0, 2]
    stds = [1, 1, 1]
    distribution_mins = [None for _ in range(len(means))]
    distribution_max = [None for _ in range(len(means))]

    hd = DistributionMixture.build_gaussian_mixture(
        distribution_amplitudes, means, stds, distribution_mins, distribution_max,
        use_quantized_distributions=True
    )

    samples = hd_rvs_many(hd)

    samples_median = np.median(samples)
    assert -0.5 < samples_median < 0.5
    samples_std = np.std(samples)
    assert 1 < samples_std < 4
    assert abs(hd.pdf(-2.) - 0.147917229965673) < 1e-6
    assert abs(hd.pdf(-1.5) - 0.) < 1e-6
    assert abs(hd.pdf(1.) - 0.16314590372033272) < 1e-6
    assert abs(hd.pdf(5.) - 0.001993471656810324) < 1e-6
    assert abs(hd.cdf(-2.) - 0.2528340972073022) < 1e-6
    assert abs(hd.cdf(-1.5) - 0.2528340972073022) < 1e-6
    assert abs(hd.cdf(1.) - 0.7471659027926978) < 1e-6
    assert abs(hd.cdf(5.) - 0.99992245064379) < 1e-6

    assert hd.min() == -np.inf
    assert hd.max() == np.inf
    assert abs(hd.mean() - 0.) < 1e-6
    assert abs(hd.var() - 3.749999989027785) < 1e-6
    assert abs(hd.std() - 1.9364916702706947) < 1e-6
    # Verify that hd mean and variance also correspond to mean and variance of sampling.
    assert abs(hd.mean() - np.mean(samples)) < 1e-1
    assert abs((hd.var() - np.var(samples)) / hd.var()) < 1e-1
