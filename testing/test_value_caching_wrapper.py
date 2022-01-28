"""
Tests for  pipelines
========================================

..
    Copyright 2021, Neuraxio Inc.

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

from sklearn.utils._testing import ignore_warnings
from typing import Callable
import numpy as np
import pytest
import time
import shutil
import uuid
import os
import tempfile

from neuraxle.base import CX, Identity, StepWithContext
from neuraxle.pipeline import Pipeline
from neuraxle.steps.misc import TapeCallbackFunction, FitTransformCallbackStep
from neuraxle.steps.caching import JoblibTransformDIValueCachingWrapper


class PipelineTestCase:
    def __init__(
        self, tape, data_inputs, expected_outputs, steps,
        expected_tape_fit,
        expected_tape_transform,
        expected_tape_fit_transform,
    ):
        self.transform_tape = tape
        self.data_inputs = data_inputs
        self.expected_outputs = expected_outputs
        self.steps = steps

        self.expected_transform_tape_when_fit = expected_tape_fit
        self.expected_transform_tape_when_transform = expected_tape_transform
        self.expected_transform_tape_when_fit_transform = expected_tape_fit_transform


def create_test_cases():
    data_inputs = np.ones((1, 1))
    expected_outputs = np.ones((1, 1))

    def tape_0_without_checkpoint_test_arguments():
        tape = TapeCallbackFunction()
        tape_fit = TapeCallbackFunction()
        tape_without_checkpoint_test_arguments = PipelineTestCase(
            tape,
            data_inputs,
            expected_outputs,
            [
                ("a", FitTransformCallbackStep(tape.callback, tape_fit.callback, ["1"])),
                ("b", FitTransformCallbackStep(tape.callback, tape_fit.callback, ["2"])),
                ("c", FitTransformCallbackStep(tape.callback, tape_fit.callback, ["3"]))
            ],
            ["1", "2"],
            ["1", "2", "3"],
            ["1", "2", "3"]
        )
        return tape_without_checkpoint_test_arguments

    def tape_1_checkpoint_not_saved_test_arguments():
        tape2 = TapeCallbackFunction()
        tape2_fit = TapeCallbackFunction()
        tape_checkpoint_not_saved_test_arguments = PipelineTestCase(
            tape2,
            data_inputs,
            expected_outputs,
            [
                ("a", JoblibTransformDIValueCachingWrapper(Identity())),
                ("b", FitTransformCallbackStep(tape2.callback, tape2_fit.callback, ["1"])),
                ("c", FitTransformCallbackStep(tape2.callback, tape2_fit.callback, ["2"])),
                ("d", FitTransformCallbackStep(tape2.callback, tape2_fit.callback, ["3"]))
            ],
            ["1", "2"],
            ["1", "2", "3"],
            ["1", "2", "3"]
        )
        return tape_checkpoint_not_saved_test_arguments

    def tape_2_checkpoint_saved_after_first_step_test_arguments():
        tape3 = TapeCallbackFunction()
        tape3_fit = TapeCallbackFunction()
        tape_checkpoint_saved_after_first_step_test_arguments = PipelineTestCase(
            tape3,
            data_inputs,
            expected_outputs,
            [
                ("a", JoblibTransformDIValueCachingWrapper(
                    FitTransformCallbackStep(tape3.callback, tape3_fit.callback, ["1"]))),
                ("b", FitTransformCallbackStep(tape3.callback, tape3_fit.callback, ["2"])),
                ("c", FitTransformCallbackStep(tape3.callback, tape3_fit.callback, ["3"]))
            ],
            ["1", "2"],
            ["1", "2", "3"],
            ["2", "3"]
        )
        return tape_checkpoint_saved_after_first_step_test_arguments

    def tape_3_checkpoint_saved_after_second_step_test_arguments():
        tape4 = TapeCallbackFunction()
        tape4_fit = TapeCallbackFunction()
        tape_checkpoint_saved_after_second_step_test_arguments = PipelineTestCase(
            tape4,
            data_inputs,
            expected_outputs,
            [
                ("a", FitTransformCallbackStep(tape4.callback, tape4_fit.callback, ["1"])),
                ("b", JoblibTransformDIValueCachingWrapper(
                    FitTransformCallbackStep(tape4.callback, tape4_fit.callback, ["2"]))),
                ("c", FitTransformCallbackStep(tape4.callback, tape4_fit.callback, ["3"]))
            ],
            ["1", "2"],
            ["1", "2", "3"],
            ["1", "3"]
        )
        return tape_checkpoint_saved_after_second_step_test_arguments

    def tape_4_checkpoint_saved_after_last_step_test_arguments():
        tape5 = TapeCallbackFunction()
        tape5_fit = TapeCallbackFunction()
        tape_checkpoint_saved_after_last_step_test_arguments = PipelineTestCase(
            tape5,
            data_inputs,
            expected_outputs,
            [
                ("a", FitTransformCallbackStep(tape5.callback, tape5_fit.callback, ["1"])),
                ("b", FitTransformCallbackStep(tape5.callback, tape5_fit.callback, ["2"])),
                ("c", JoblibTransformDIValueCachingWrapper(
                    FitTransformCallbackStep(tape5.callback, tape5_fit.callback, ["3"])))
            ],
            ["1", "2"],
            ["1", "2", "3"],
            ["1", "2", "3"]  # transform checkpoint for #3 failed during fit as it was not transformed.
        )
        return tape_checkpoint_saved_after_last_step_test_arguments

    def tape_5_checkpoint_saved_inside_subpipeline_last_step():
        tape6 = TapeCallbackFunction()
        tape6_fit = TapeCallbackFunction()
        tape_checkpoint_saved_inside_subpipeline_last_step = PipelineTestCase(
            tape6,
            data_inputs,
            expected_outputs,
            [
                Pipeline([
                    ("a", JoblibTransformDIValueCachingWrapper(FitTransformCallbackStep(
                        tape6.callback, tape6_fit.callback, ["1"]))),
                    ("b", JoblibTransformDIValueCachingWrapper(FitTransformCallbackStep(
                        tape6.callback, tape6_fit.callback, ["2"])))
                ]),
                ("c", FitTransformCallbackStep(tape6.callback, tape6_fit.callback, ["3"])),
                ("d", FitTransformCallbackStep(tape6.callback, tape6_fit.callback, ["4"])),
            ],
            ["1", "2", "3"],
            ["1", "2", "3", "4"],
            ["3", "4"],
        )
        return tape_checkpoint_saved_inside_subpipeline_last_step

    def tape_6_saved_checkpoint_after_another_saved_checkpoint():
        tape10 = TapeCallbackFunction()
        tape10_fit = TapeCallbackFunction()
        tape_saved_checkpoint_after_another_saved_checkpoint = PipelineTestCase(
            tape10,
            data_inputs,
            expected_outputs,
            [
                ("a", FitTransformCallbackStep(tape10.callback, tape10_fit.callback, ["1"])),
                ("b", JoblibTransformDIValueCachingWrapper(FitTransformCallbackStep(
                    tape10.callback, tape10_fit.callback, ["2"]))),
                ("c", FitTransformCallbackStep(tape10.callback, tape10_fit.callback, ["3"])),
                ("d", JoblibTransformDIValueCachingWrapper(FitTransformCallbackStep(
                    tape10.callback, tape10_fit.callback, ["4"]))),
                ("e", FitTransformCallbackStep(tape10.callback, tape10_fit.callback, ["5"]))
            ],
            ["1", "2", "3", "4"],
            ["1", "2", "3", "4", "5"],
            ["1", "3", "5"],
        )
        return tape_saved_checkpoint_after_another_saved_checkpoint

    def tape_7_multiple_checkpoint_in_a_row():
        tape11 = TapeCallbackFunction()
        tape11_fit = TapeCallbackFunction()
        tape_multiple_checkpoint_in_a_row = PipelineTestCase(
            tape11,
            data_inputs,
            expected_outputs,
            [
                ("a", FitTransformCallbackStep(tape11.callback, tape11_fit.callback, ["1"])),
                ("b", FitTransformCallbackStep(tape11.callback, tape11_fit.callback, ["2"])),
                ("c", FitTransformCallbackStep(tape11.callback, tape11_fit.callback, ["3"])),
                ("d", JoblibTransformDIValueCachingWrapper(Identity()))
            ],
            ["1", "2", "3"],
            ["1", "2", "3"],
            ["1", "2", "3"]
        )
        return tape_multiple_checkpoint_in_a_row

    return [
        tape_0_without_checkpoint_test_arguments,
        tape_1_checkpoint_not_saved_test_arguments,
        tape_2_checkpoint_saved_after_first_step_test_arguments,
        tape_3_checkpoint_saved_after_second_step_test_arguments,
        tape_4_checkpoint_saved_after_last_step_test_arguments,
        tape_5_checkpoint_saved_inside_subpipeline_last_step,
        tape_6_saved_checkpoint_after_another_saved_checkpoint,
        tape_7_multiple_checkpoint_in_a_row
    ]


@pytest.mark.parametrize("test_case", create_test_cases())
def test_1_fit_value_caching(test_case: Callable):
    test_case: PipelineTestCase = test_case()
    pipeline = Pipeline(test_case.steps)

    pipeline = pipeline.fit(test_case.data_inputs, test_case.expected_outputs)

    actual_tape = test_case.transform_tape.get_name_tape()
    assert actual_tape == test_case.expected_transform_tape_when_fit


@pytest.mark.parametrize("test_case", create_test_cases())
def test_2_transform_value_caching(test_case: Callable):
    test_case: PipelineTestCase = test_case()
    pipeline = Pipeline(test_case.steps).with_context(CX())

    actual_data_inputs = pipeline.transform(test_case.data_inputs)

    actual_tape = test_case.transform_tape.get_name_tape()
    assert actual_tape == test_case.expected_transform_tape_when_transform
    assert np.array_equal(actual_data_inputs, test_case.data_inputs)


@pytest.mark.parametrize("test_case", create_test_cases())
def test_3_fit_transform_value_caching(test_case: Callable):
    test_case: PipelineTestCase = test_case()
    pipeline = Pipeline(test_case.steps)

    pipeline = pipeline.fit(test_case.data_inputs, test_case.expected_outputs)
    actual_data_inputs = pipeline.transform(test_case.data_inputs)

    actual_tape = test_case.transform_tape.get_name_tape()
    assert actual_tape == test_case.expected_transform_tape_when_fit + test_case.expected_transform_tape_when_fit_transform
    assert np.array_equal(actual_data_inputs, test_case.data_inputs)
