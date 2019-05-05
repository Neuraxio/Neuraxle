# Copyright 2019, The Neuraxle Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pprint import pprint

import pytest

from neuraxle.hyperparams.conversion import flat_to_dict, dict_to_flat
from neuraxle.typing import FlatHyperparams, DictHyperparams

hyperparams_flat_and_dict_pairs = [
    # Pair 1:
    ({
         "a__learning_rate": 7
     }, {
         "a": {
             "learning_rate": 7
         }
     }),
    # Pair 2:
    ({
         "b__a__learning_rate": 7,
         "b__learning_rate": 9
     }, {
         "b": {
             "a": {
                 "learning_rate": 7
             },
             "learning_rate": 9
         }
     }),
]


@pytest.mark.parametrize("flat,expected_dic", hyperparams_flat_and_dict_pairs)
def test_flat_to_dict_hyperparams(flat: FlatHyperparams, expected_dic: DictHyperparams):
    dic = flat_to_dict(flat)

    assert dict(dic) == dict(expected_dic)


@pytest.mark.parametrize("expected_flat,dic", hyperparams_flat_and_dict_pairs)
def test_dict_to_flat_hyperparams(expected_flat: FlatHyperparams, dic: DictHyperparams):
    flat = dict_to_flat(dic)

    pprint(dict(flat))
    pprint(expected_flat)
    assert dict(flat) == dict(expected_flat)
