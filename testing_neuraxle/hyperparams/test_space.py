"""
Tests for Hyperparameters Distribution Spaces
=============================================

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
import copy
from collections import OrderedDict

import pytest
import scipy
from neuraxle.hyperparams.distributions import (Boolean, Choice,
                                                FixedHyperparameter,
                                                HyperparameterDistribution,
                                                LogNormal, LogUniform, Normal,
                                                PriorityChoice, Quantized,
                                                RandInt, Uniform)
from neuraxle.hyperparams.scipy_distributions import Gaussian, Poisson
from neuraxle.hyperparams.space import (FlatDict, HyperparameterSamples,
                                        HyperparameterSpace, RecursiveDict)

HYPERPARAMS_FLAT_AND_DICT_PAIRS = [(
    # Pair 1:
    {
        "a__learning_rate": 7
    },
    {
        "a": {
            "learning_rate": 7
        }
    }),
    # Pair 2:
    ({
        "b__a__learning_rate": 7,
        "b__learning_rate": 9
    },
    {
        "b": {
            "a": {
                "learning_rate": 7
            },
            "learning_rate": 9
        }
    }),
]


@pytest.mark.parametrize("class_to_test", [RecursiveDict, HyperparameterSamples])
@pytest.mark.parametrize("flat,expected_dic", HYPERPARAMS_FLAT_AND_DICT_PAIRS)
def test_flat_to_dict_hyperparams(flat: dict, expected_dic: dict, class_to_test):
    from_flat_dic = class_to_test(flat)
    from_nested_dic = class_to_test(expected_dic)

    assert from_flat_dic == from_nested_dic
    assert from_flat_dic.to_flat_dict() == flat
    assert from_nested_dic.to_flat_dict() == flat
    assert from_nested_dic.to_nested_dict() == expected_dic
    assert from_flat_dic.to_nested_dict() == expected_dic


HYPE_SPACE = HyperparameterSpace(OrderedDict({
    "a__b__c": PriorityChoice([0, 1, False, "Test"]),
    "a__b__q__c": Quantized(Uniform(-10, 10)),
    "a__b__q__q": Quantized(Uniform(-10, 10)),
    "a__c": Choice([0, 1, False, "Test"]),
    "a__e__q__c": Choice([0, 1, False, "Test"]),
    "a__test": Boolean(),
    "d__param": RandInt(-10, 10),
    "d__u": Uniform(-10, 10),
    "e__alpha": Normal(0.0, 1.0),
    "e__f__g": LogNormal(0.0, 2.0),
    "e__other": LogUniform(0.001, 10),
    "p__could_also_be_as_fixed": FixedHyperparameter("also hey"),
    "scipy__gaussian": Gaussian(-1, 1),
    "scipy__poisson": Poisson(1.0, 2.0),
    "scipy__scipy__gaussian": scipy.stats.randint(0, 10)
}))


def test_hyperparams_space_rvs_outputs_samples():
    space = copy.deepcopy(HYPE_SPACE)

    samples = space.rvs()

    assert isinstance(samples, HyperparameterSamples)
    assert len(samples) == len(space)
    for k, v in samples.iter_flat():
        assert k in space
        assert not isinstance(v, HyperparameterDistribution)


@pytest.mark.parametrize("hd", list(HYPE_SPACE.to_flat_dict().values()))
def test_hyperparams_space_rvs_outputs_in_range(hd: HyperparameterDistribution):
    for _ in range(20):

        sample = hd.rvs()

        assert sample in hd


def test_wildcards():
    EXPECTED_WILDCARDS = [
        "*b__c",
        "*b*c",
        "*q",
        "a__c",
        "*e*c",
        "*test",
        "*param",
        "*u",
        "*alpha",
        "*g",
        "*other",
        "*could_also_be_as_fixed",
        "scipy__gaussian",
        "*poisson",
        "*scipy__gaussian",
    ]

    wildcards: FlatDict = HYPE_SPACE.to_wildcards()

    for wc, ewc in zip(wildcards.keys(), EXPECTED_WILDCARDS):
        assert wc == ewc, f"{wc} != {ewc}, but should be equal as expected."
    for wv, ewv in zip(wildcards.values(), HYPE_SPACE.to_flat_dict().values()):
        assert wv == ewv, f"{str(wv)} != {str(ewv)}, but should remain the same."
