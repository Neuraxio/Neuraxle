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

from pprint import pprint

import pytest

from neuraxle.hyperparams.distributions import *
from neuraxle.hyperparams.space import flat_to_nested_dict, nested_dict_to_flat, HyperparameterSpace, \
    HyperparameterSamples

hyperparams_flat_and_dict_pairs = [
    # Pair 1:
    ({
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


@pytest.mark.parametrize("flat,expected_dic", hyperparams_flat_and_dict_pairs)
def test_flat_to_dict_hyperparams(flat: dict, expected_dic: dict):
    dic = flat_to_nested_dict(flat)

    assert dict(dic) == dict(expected_dic)


@pytest.mark.parametrize("expected_flat,dic", hyperparams_flat_and_dict_pairs)
def test_dict_to_flat_hyperparams(expected_flat: dict, dic: dict):
    flat = nested_dict_to_flat(dic)

    pprint(dict(flat))
    pprint(expected_flat)
    assert dict(flat) == dict(expected_flat)


@pytest.mark.parametrize("flat,expected_dic", hyperparams_flat_and_dict_pairs)
def test_flat_to_dict_hyperparams_with_hyperparameter_space(flat: dict, expected_dic: dict):
    dic = HyperparameterSpace(flat).to_nested_dict_as_dict_primitive()

    assert dict(dic) == dict(expected_dic)


@pytest.mark.parametrize("expected_flat,dic", hyperparams_flat_and_dict_pairs)
def test_dict_to_flat_hyperparams_with_hyperparameter_space(expected_flat: dict, dic: dict):
    flat = HyperparameterSpace(dic).to_flat_as_dict_primitive()

    pprint(dict(flat))
    pprint(expected_flat)
    assert dict(flat) == dict(expected_flat)


HYPE_SPACE = HyperparameterSpace({
    "a__test": Boolean(),
    "a__lr": Choice([0, 1, False, "Test"]),
    "a__b__c": PriorityChoice([0, 1, False, "Test"]),
    "a__b__q": Quantized(Uniform(-10, 10)),
    "d__param": RandInt(-10, 10),
    "d__u": Uniform(-10, 10),
    "e__other": LogUniform(0.001, 10),
    "e__alpha": Normal(0.0, 1.0),
    "e__f__g": LogNormal(0.0, 2.0),
    "p__other_nondistribution_params": "hey",
    "p__could_also_be_as_fixed": FixedHyperparameter("also hey"),
    "p__its_over_9k": 9001
})


@pytest.mark.parametrize("to_flat_func_name", [
    "to_flat",
    "to_flat_as_dict_primitive",
    "to_flat_as_ordered_dict_primitive"])
@pytest.mark.parametrize("to_nested_dict_func_name", [
    "to_nested_dict",
    "to_nested_dict_as_dict_primitive",
    "to_nested_dict_as_ordered_dict_primitive"])
def test_hyperparams_space_round_robin(to_nested_dict_func_name, to_flat_func_name):
    orig_space = copy.deepcopy(HYPE_SPACE)
    print(orig_space.keys())

    nestened = HyperparameterSpace(getattr(
        orig_space,
        to_nested_dict_func_name
    )())
    print(nestened)
    flattened = HyperparameterSpace(getattr(
        nestened,
        to_flat_func_name
    )())

    print(flattened.keys())
    assert flattened.to_flat_as_dict_primitive() == orig_space.to_flat_as_dict_primitive()


def test_hyperparams_space_rvs_outputs_samples():
    space = copy.deepcopy(HYPE_SPACE).to_flat()

    samples = space.rvs()

    assert isinstance(samples, HyperparameterSamples)
    assert len(samples) == len(space)
    for k, v in samples.items():
        assert k in space
        assert not isinstance(v, HyperparameterDistribution)
