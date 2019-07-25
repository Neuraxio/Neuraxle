"""
Tests for Metaopt
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
import numpy as np
from neuraxle.metaopt.random import WalkForwardTimeSeriesCrossValidation


def test_walkforward_crossvalidation_split():
    # set random seed
    np.random.seed(10)
    data_inputs = np.random.randint(0, high=1, size=(7, 20, 2)).astype(np.float)
    expected_outputs = np.random.randint(0, high=1, size=(7, 20, 2)).astype(np.float)
    step = WalkForwardTimeSeriesCrossValidation(window_size=3, initial_window_number=3, window_delay_size=2)
    train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = \
        step.split(data_inputs, expected_outputs)

    assert (len(train_data_inputs), len(train_expected_outputs),
            len(validation_data_inputs), len(validation_expected_outputs)
            ) == (3, 3, 3, 3)

    assert train_data_inputs[0].shape == (7, 9, 2)
    assert train_expected_outputs[0].shape == (7, 9, 2)
    assert validation_data_inputs[0].shape == (7, 3, 2)
    assert validation_expected_outputs[0].shape == (7, 3, 2)