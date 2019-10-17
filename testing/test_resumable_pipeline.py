"""
Tests for resumable pipelines
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

import numpy as np
import pytest

from neuraxle.base import DataContainer, ExecutionContext
from neuraxle.checkpoints import BaseCheckpointStep
from neuraxle.pipeline import Pipeline, ResumablePipeline
from neuraxle.steps.util import FitTransformCallbackStep, TapeCallbackFunction
from neuraxle.steps.misc import TapeCallbackFunction


class SomeCheckpointStep(BaseCheckpointStep):
    def __init__(self, data_container: DataContainer = None):
        super().__init__()
        self.saved = False
        self.saved_data_container = data_container
        self.checkpoint_path = None

    def set_checkpoint_path(self, path):
        self.checkpoint_path = path
        pass

    def read_checkpoint(self, data_container: DataContainer):
        return self.saved_data_container

    def save_checkpoint(self, data_container: DataContainer):
        self.saved_data_container = data_container
        self.saved = True
        return data_container

    def should_resume(self, data_container, context: ExecutionContext) -> bool:
        return self.saved_data_container is not None


class ResumablePipelineTestCase:
    def __init__(self, tape, data_inputs, expected_outputs, steps, expected_tape):
        self.steps = steps
        self.expected_outputs = expected_outputs
        self.expected_tape = expected_tape
        self.data_inputs = data_inputs
        self.tape = tape


def create_test_case(di, eo, steps, expected_tape):
    tape = TapeCallbackFunction()
    return ResumablePipelineTestCase(tape, di, eo, steps, expected_tape)


def create_test_cases():
    data_inputs = np.ones((1, 1))
    expected_outputs = np.ones((1, 1))
    dc = DataContainer(current_ids=range(len(data_inputs)), data_inputs=data_inputs, expected_outputs=expected_outputs)

    tape = TapeCallbackFunction()
    tape_fit = TapeCallbackFunction()
    tape_without_checkpoint_test_arguments = ResumablePipelineTestCase(
        tape,
        data_inputs,
        expected_outputs,
        [
            ("a", FitTransformCallbackStep(tape.callback, tape_fit.callback, ["1"])),
            ("b", FitTransformCallbackStep(tape.callback, tape_fit.callback, ["2"])),
            ("c", FitTransformCallbackStep(tape.callback, tape_fit.callback, ["3"]))
        ],
        ["1", "2", "3"])

    tape2 = TapeCallbackFunction()
    tape2_fit = TapeCallbackFunction()
    tape_checkpoint_not_saved_test_arguments = ResumablePipelineTestCase(
        tape2,
        data_inputs,
        expected_outputs,
        [
            ("a", FitTransformCallbackStep(tape2.callback, tape2_fit.callback, ["1"])),
            ("b", SomeCheckpointStep(data_container=None)
             ),
            ("c", FitTransformCallbackStep(tape2.callback, tape2_fit.callback, ["2"])),
            ("d", FitTransformCallbackStep(tape2.callback, tape2_fit.callback, ["3"]))
        ],
        ["1", "2", "3"])

    tape3 = TapeCallbackFunction()
    tape3_fit = TapeCallbackFunction()
    tape_checkpoint_saved_after_first_step_test_arguments = ResumablePipelineTestCase(
        tape3,
        data_inputs,
        expected_outputs,
        [
            ("a", FitTransformCallbackStep(tape3.callback, tape3_fit.callback, ["1"])),
            ("b", SomeCheckpointStep(data_container=dc)
             ),
            ("c", FitTransformCallbackStep(tape3.callback, tape3_fit.callback, ["2"])),
            ("d", FitTransformCallbackStep(tape3.callback, tape3_fit.callback, ["3"]))
        ],
        ["2", "3"])

    tape4 = TapeCallbackFunction()
    tape4_fit = TapeCallbackFunction()
    tape_checkpoint_saved_after_second_step_test_arguments = ResumablePipelineTestCase(
        tape4,
        data_inputs,
        expected_outputs,
        [
            ("a", FitTransformCallbackStep(tape4.callback, tape4_fit.callback, ["1"])),
            ("b", FitTransformCallbackStep(tape4.callback, tape4_fit.callback, ["2"])),
            ("c", SomeCheckpointStep(data_container=dc)
             ),
            ("d", FitTransformCallbackStep(tape4.callback, tape4_fit.callback, ["3"]))
        ],
        ["3"])

    tape5 = TapeCallbackFunction()
    tape5_fit = TapeCallbackFunction()
    tape_checkpoint_saved_after_last_step_test_arguments = ResumablePipelineTestCase(
        tape5,
        data_inputs,
        expected_outputs,
        [
            ("a", FitTransformCallbackStep(tape5.callback, tape5_fit.callback, ["1"])),
            ("b", FitTransformCallbackStep(tape5.callback, tape5_fit.callback, ["2"])),
            ("c", FitTransformCallbackStep(tape5.callback, tape5_fit.callback, ["3"])),
            ("d", SomeCheckpointStep(data_container=dc)
             ),
        ],
        [])

    tape6 = TapeCallbackFunction()
    tape6_fit = TapeCallbackFunction()
    tape_checkpoint_saved_inside_subpipeline_last_step = ResumablePipelineTestCase(
        tape6,
        data_inputs,
        expected_outputs,
        [
            ("a", FitTransformCallbackStep(tape6.callback, tape6_fit.callback, ["1"])),
            ResumablePipeline([
                ("b", FitTransformCallbackStep(tape6.callback, tape6_fit.callback, ["2"])),
                ("d", SomeCheckpointStep(data_container=dc)
                 ),
            ]),
            ("e", FitTransformCallbackStep(tape6.callback, tape6_fit.callback, ["3"])),
            ("f", FitTransformCallbackStep(tape6.callback, tape6_fit.callback, ["4"])),
        ],
        ["3", "4"])

    tape7 = TapeCallbackFunction()
    tape7_fit = TapeCallbackFunction()
    tape_checkpoint_saved_inside_subpipeline_first_step = ResumablePipelineTestCase(
        tape7,
        data_inputs,
        expected_outputs,
        [
            ("a", FitTransformCallbackStep(tape7.callback, tape7_fit.callback, ["1"])),
            ResumablePipeline([
                ("d", SomeCheckpointStep(data_container=dc)
                 ),
                ("b", FitTransformCallbackStep(tape7.callback, tape7_fit.callback, ["2"])),
            ]),
            ("e", FitTransformCallbackStep(tape7.callback, tape7_fit.callback, ["3"])),
            ("f", FitTransformCallbackStep(tape7.callback, tape7_fit.callback, ["4"])),
        ],
        ["2", "3", "4"])

    tape8 = TapeCallbackFunction()
    tape8_fit = TapeCallbackFunction()
    tape_checkpoint_saved_inside_subpipeline_step_in_the_middle = ResumablePipelineTestCase(
        tape8,
        data_inputs,
        expected_outputs,
        [
            ("a", FitTransformCallbackStep(tape8.callback, tape8_fit.callback, ["1"])),
            ResumablePipeline([
                ("b", FitTransformCallbackStep(tape8.callback, tape8_fit.callback, ["2"])),
                ("d", SomeCheckpointStep(data_container=dc)
                 ),
                ("e", FitTransformCallbackStep(tape8.callback, tape8_fit.callback, ["3"])),
            ]),
            ("f", FitTransformCallbackStep(tape8.callback, tape8_fit.callback, ["4"])),
        ],
        ["3", "4"])

    tape9 = TapeCallbackFunction()
    tape9_fit = TapeCallbackFunction()
    tape_checkpoint_saved_inside_subpipeline_of_subpipeline = ResumablePipelineTestCase(
        tape9,
        data_inputs,
        expected_outputs,
        [
            ("a", FitTransformCallbackStep(tape9.callback, tape9_fit.callback, ["1"])),
            ResumablePipeline([
                ("b", FitTransformCallbackStep(tape9.callback, tape9_fit.callback, ["2"])),
                ResumablePipeline([
                    ("e", FitTransformCallbackStep(tape9.callback, tape9_fit.callback, ["3"])),
                    ("d", SomeCheckpointStep(data_container=dc)
                     ),
                    ("f", FitTransformCallbackStep(tape9.callback, tape9_fit.callback, ["4"])),
                ]),
                ("g", FitTransformCallbackStep(tape9.callback, tape9_fit.callback, ["5"])),
            ]),
            ("h", FitTransformCallbackStep(tape9.callback, tape9_fit.callback, ["6"])),
        ],
        ["4", "5", "6"])

    tape10 = TapeCallbackFunction()
    tape10_fit = TapeCallbackFunction()
    tape_saved_checkpoint_after_another_saved_checkpoint = ResumablePipelineTestCase(
        tape10,
        data_inputs,
        expected_outputs,
        [
            ("a", FitTransformCallbackStep(tape10.callback, tape10_fit.callback, ["1"])),
            ("b", SomeCheckpointStep(data_container=dc)),
            ("c", FitTransformCallbackStep(tape10.callback, tape10_fit.callback, ["2"])),
            ("b", SomeCheckpointStep(data_container=dc)),
            ("d", FitTransformCallbackStep(tape10.callback, tape10_fit.callback, ["3"]))
        ],
        ["3"])

    tape11 = TapeCallbackFunction()
    tape11_fit = TapeCallbackFunction()
    tape_multiple_checkpoint_in_a_row = ResumablePipelineTestCase(
        tape11,
        data_inputs,
        expected_outputs,
        [
            ("a", FitTransformCallbackStep(tape11.callback, tape11_fit.callback, ["1"])),
            ("pickle_1", SomeCheckpointStep(data_container=dc)),
            ("pickle_2", SomeCheckpointStep(data_container=dc)),
            ("c", FitTransformCallbackStep(tape11.callback, tape11_fit.callback, ["2"])),
            ("d", FitTransformCallbackStep(tape11.callback, tape11_fit.callback, ["3"]))
        ],
        ["2", "3"])

    return [
        tape_without_checkpoint_test_arguments,
        tape_checkpoint_not_saved_test_arguments,
        tape_checkpoint_saved_after_first_step_test_arguments,
        tape_checkpoint_saved_after_second_step_test_arguments,
        tape_checkpoint_saved_after_last_step_test_arguments,
        tape_checkpoint_saved_inside_subpipeline_first_step,
        tape_checkpoint_saved_inside_subpipeline_last_step,
        tape_checkpoint_saved_inside_subpipeline_step_in_the_middle,
        tape_checkpoint_saved_inside_subpipeline_of_subpipeline,
        tape_saved_checkpoint_after_another_saved_checkpoint,
        tape_multiple_checkpoint_in_a_row
    ]


@pytest.mark.parametrize("test_case", create_test_cases())
def test_should_fit_transform_each_steps(test_case: ResumablePipelineTestCase, tmpdir):
    pipeline = ResumablePipeline(
        steps=test_case.steps,
        cache_folder=tmpdir
    )

    actual_pipeline, actual_data_inputs = pipeline.fit_transform(test_case.data_inputs, test_case.expected_outputs)

    actual_tape = test_case.tape.get_name_tape()
    assert isinstance(actual_pipeline, Pipeline)
    assert actual_tape == test_case.expected_tape
    assert np.array_equal(actual_data_inputs, test_case.data_inputs)


@pytest.mark.parametrize("test_case", create_test_cases())
def test_should_fit_each_steps(test_case: ResumablePipelineTestCase, tmpdir):
    pipeline = ResumablePipeline(
        steps=test_case.steps,
        cache_folder=tmpdir
    )

    actual_pipeline = pipeline.fit(test_case.data_inputs, test_case.expected_outputs)

    actual_tape = test_case.tape.get_name_tape()
    assert isinstance(actual_pipeline, Pipeline)
    assert actual_tape == test_case.expected_tape[:-1]


@pytest.mark.parametrize("test_case", create_test_cases())
def test_should_transform_each_steps(test_case: ResumablePipelineTestCase, tmpdir):
    pipeline = ResumablePipeline(
        steps=test_case.steps,
        cache_folder=tmpdir
    )

    actual_data_inputs = pipeline.transform(test_case.data_inputs)

    actual_tape = test_case.tape.get_name_tape()
    assert actual_tape == test_case.expected_tape
    assert np.array_equal(actual_data_inputs, test_case.data_inputs)
