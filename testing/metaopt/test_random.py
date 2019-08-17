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
import pytest
import math
import numpy as np
from neuraxle.metaopt.random import WalkForwardTimeSeriesCrossValidation, AnchoredWalkForwardTimeSeriesCrossValidation

classic_walforward_parameters = {
    # (training_size, validation_size, padding_between_training_and_validation, drop_remainder)

    # Pair 1:
    (9, 3, 1, False),
    # Pair 2:
    (9, 3, 1, True),
    # Pair 3:
    (9, 3, 2, False),
    # Pair 4:
    (8, 3, 2, False),
    # Pair 4:
    (5, 2, 3, True),
    # Pair 5 (Default Parameters):
    (2, None, 0, False),
}


@pytest.mark.parametrize("training_window_size, validation_window_size, "
                         "padding_between_training_and_validation, drop_remainder",
                         classic_walforward_parameters)
def test_classic_walkforward_crossvalidation_split(training_window_size: int, validation_window_size: int,
                                                   padding_between_training_and_validation: int, drop_remainder: bool):
    # Arrange
    # set random seed
    np.random.seed(10)

    # Defines the data shape.
    batch_size = 7
    time_series_size = 20
    features_size = 2

    validation_window_size_temp = validation_window_size

    if validation_window_size is None:
        validation_window_size_temp = training_window_size

    if drop_remainder:
        # We have one less number of fold if we drop remainder.
        number_of_fold = math.floor(
            (time_series_size - training_window_size - padding_between_training_and_validation) /
            validation_window_size_temp
        )
    else:
        number_of_fold = math.ceil(
            (time_series_size - training_window_size - padding_between_training_and_validation) /
            validation_window_size_temp
        )

    # Calculate the size of the remainder
    remainder_size = (time_series_size - training_window_size - padding_between_training_and_validation) % \
                     validation_window_size_temp
    if remainder_size == 0:
        # Remainder of 0 means the data perfecftly fits in the number of fold and remainder should window_size instead.
        remainder_size = validation_window_size_temp

    # Initialize the inputs.
    data_inputs = np.random.randint(low=0, high=1, size=(batch_size, time_series_size, features_size)).astype(np.float)
    expected_outputs = np.random.randint(low=0, high=1, size=(batch_size, time_series_size, features_size)) \
        .astype(np.float)

    # Initialize the class to test.
    step = WalkForwardTimeSeriesCrossValidation(
        validation_window_size=validation_window_size,
        training_window_size=training_window_size,
        padding_between_training_and_validation=padding_between_training_and_validation,
        drop_remainder=drop_remainder
    )

    # Act
    train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = \
        step.split(data_inputs, expected_outputs)

    # Assert
    assert len(train_data_inputs) == number_of_fold
    assert len(train_expected_outputs) == number_of_fold
    assert len(validation_data_inputs) == number_of_fold
    assert len(validation_expected_outputs) == number_of_fold

    assert train_data_inputs[0].shape == (batch_size, training_window_size, features_size)
    assert train_expected_outputs[0].shape == (batch_size, training_window_size, features_size)
    assert validation_data_inputs[0].shape == (batch_size, validation_window_size_temp, features_size)
    assert validation_expected_outputs[0].shape == (batch_size, validation_window_size_temp, features_size)

    assert train_data_inputs[1].shape == (batch_size, training_window_size, features_size)
    assert train_expected_outputs[1].shape == (batch_size, training_window_size, features_size)
    assert validation_data_inputs[1].shape == (batch_size, validation_window_size_temp, features_size)
    assert validation_expected_outputs[1].shape == (batch_size, validation_window_size_temp, features_size)

    if drop_remainder:
        assert train_data_inputs[-1].shape == (
            batch_size, training_window_size, features_size)
        assert train_expected_outputs[-1].shape == (
            batch_size, training_window_size, features_size)
        assert validation_data_inputs[-1].shape == (batch_size, validation_window_size_temp, features_size)
        assert validation_expected_outputs[-1].shape == (batch_size, validation_window_size_temp, features_size)
    else:
        assert train_data_inputs[-1].shape == (
            batch_size, training_window_size, features_size)
        assert train_expected_outputs[-1].shape == (
            batch_size, training_window_size, features_size)
        assert validation_data_inputs[-1].shape == (batch_size, remainder_size, features_size)
        assert validation_expected_outputs[-1].shape == (batch_size, remainder_size, features_size)


anchored_walforward_parameters = {
    # (minimum_training_size, validation_window_size, padding_between_training_and_validation, drop_remainder)

    # Pair 1:
    (9, 3, 1, False),
    # Pair 2:
    (9, 3, 1, True),
    # Pair 3:
    (9, 3, 2, False),
    # Pair 4:
    (8, 3, 2, False),
    # Pair 4:
    (5, 2, 3, True),
    # Pair 5 (Default Parameters):
    (2, None, 0, False),
}


@pytest.mark.parametrize(
    "minimum_training_size, validation_window_size, padding_between_training_and_validation, drop_remainder",
    anchored_walforward_parameters)
def test_anchored_walkforward_crossvalidation_split(minimum_training_size: int, validation_window_size: int,
                                                    padding_between_training_and_validation: int, drop_remainder: bool):
    # Arrange
    # set random seed
    np.random.seed(10)

    # Defines the data shape.
    batch_size = 7
    time_series_size = 20
    features_size = 2

    validation_window_size_temp = validation_window_size

    if validation_window_size is None:
        validation_window_size_temp = minimum_training_size

    if drop_remainder:
        # We have one less number of fold if we drop remainder.
        number_of_fold = math.floor(
            (time_series_size - minimum_training_size - padding_between_training_and_validation) /
            validation_window_size_temp
        )
    else:
        number_of_fold = math.ceil(
            (time_series_size - minimum_training_size - padding_between_training_and_validation) /
            validation_window_size_temp
        )

    # Calculate the size of the remainder
    remainder_size = (time_series_size - minimum_training_size - padding_between_training_and_validation) % \
                     validation_window_size_temp
    if remainder_size == 0:
        # Remainder of 0 means the data perfectly fits in the number of fold and remainder should window_size instead.
        remainder_size = validation_window_size_temp

    # Initialize the inputs.
    data_inputs = np.random.randint(low=0, high=1, size=(batch_size, time_series_size, features_size)).astype(np.float)
    expected_outputs = np.random.randint(low=0, high=1, size=(batch_size, time_series_size, features_size)) \
        .astype(np.float)

    # Initialize the class to test.
    step = AnchoredWalkForwardTimeSeriesCrossValidation(
        validation_window_size=validation_window_size,
        minimum_training_size=minimum_training_size,
        padding_between_training_and_validation=padding_between_training_and_validation,
        drop_remainder=drop_remainder
    )

    # Act
    train_data_inputs, train_expected_outputs, validation_data_inputs, validation_expected_outputs = \
        step.split(data_inputs, expected_outputs)

    # Assert
    assert len(train_data_inputs) == number_of_fold
    assert len(train_expected_outputs) == number_of_fold
    assert len(validation_data_inputs) == number_of_fold
    assert len(validation_expected_outputs) == number_of_fold

    assert train_data_inputs[0].shape == (batch_size, minimum_training_size, features_size)
    assert train_expected_outputs[0].shape == (batch_size, minimum_training_size, features_size)
    assert validation_data_inputs[0].shape == (batch_size, validation_window_size_temp, features_size)
    assert validation_expected_outputs[0].shape == (batch_size, validation_window_size_temp, features_size)

    assert train_data_inputs[1].shape == (batch_size, minimum_training_size + validation_window_size_temp,
                                          features_size)
    assert train_expected_outputs[1].shape == (batch_size, minimum_training_size + validation_window_size_temp,
                                               features_size)
    assert validation_data_inputs[1].shape == (batch_size, validation_window_size_temp, features_size)
    assert validation_expected_outputs[1].shape == (batch_size, validation_window_size_temp, features_size)

    if drop_remainder:
        assert train_data_inputs[-1].shape == (
            batch_size, minimum_training_size + (number_of_fold - 1) * validation_window_size_temp, features_size)
        assert train_expected_outputs[-1].shape == (
            batch_size, minimum_training_size + (number_of_fold - 1) * validation_window_size_temp, features_size)
        assert validation_data_inputs[-1].shape == (batch_size, validation_window_size_temp, features_size)
        assert validation_expected_outputs[-1].shape == (batch_size, validation_window_size_temp, features_size)
    else:
        assert train_data_inputs[-1].shape == (
            batch_size, time_series_size - remainder_size - padding_between_training_and_validation, features_size)
        assert train_expected_outputs[-1].shape == (
            batch_size, time_series_size - remainder_size - padding_between_training_and_validation, features_size)
        assert validation_data_inputs[-1].shape == (batch_size, remainder_size, features_size)
        assert validation_expected_outputs[-1].shape == (batch_size, remainder_size, features_size)
