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

from neuraxle.base import DataContainer
from neuraxle.checkpoints import BaseCheckpointStep
from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import TransformCallbackStep, TapeCallbackFunction


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

    def should_resume(self, data_container) -> bool:
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
    tape_without_checkpoint_test_arguments = ResumablePipelineTestCase(
        tape,
        data_inputs,
        expected_outputs,
        [
            ("a", TransformCallbackStep(tape.callback, ["1"])),
            ("b", TransformCallbackStep(tape.callback, ["2"])),
            ("c", TransformCallbackStep(tape.callback, ["3"]))
        ],
        ["1", "2", "3"])

    tape2 = TapeCallbackFunction()
    tape_checkpoint_not_saved_test_arguments = ResumablePipelineTestCase(
        tape2,
        data_inputs,
        expected_outputs,
        [
            ("a", TransformCallbackStep(tape2.callback, ["1"])),
            ("b", SomeCheckpointStep(data_container=None)
             ),
            ("c", TransformCallbackStep(tape2.callback, ["2"])),
            ("d", TransformCallbackStep(tape2.callback, ["3"]))
        ],
        ["1", "2", "3"])

    tape3 = TapeCallbackFunction()
    tape_checkpoint_saved_after_first_step_test_arguments = ResumablePipelineTestCase(
        tape3,
        data_inputs,
        expected_outputs,
        [
            ("a", TransformCallbackStep(tape3.callback, ["1"])),
            ("b", SomeCheckpointStep(data_container=dc)
             ),
            ("c", TransformCallbackStep(tape3.callback, ["2"])),
            ("d", TransformCallbackStep(tape3.callback, ["3"]))
        ],
        ["2", "3"])

    tape4 = TapeCallbackFunction()
    tape_checkpoint_saved_after_second_step_test_arguments = ResumablePipelineTestCase(
        tape4,
        data_inputs,
        expected_outputs,
        [
            ("a", TransformCallbackStep(tape4.callback, ["1"])),
            ("b", TransformCallbackStep(tape4.callback, ["2"])),
            ("c", SomeCheckpointStep(data_container=dc)
             ),
            ("d", TransformCallbackStep(tape4.callback, ["3"]))
        ],
        ["3"])

    tape5 = TapeCallbackFunction()
    tape_checkpoint_saved_after_last_step_test_arguments = ResumablePipelineTestCase(
        tape5,
        data_inputs,
        expected_outputs,
        [
            ("a", TransformCallbackStep(tape5.callback, ["1"])),
            ("b", TransformCallbackStep(tape5.callback, ["2"])),
            ("c", TransformCallbackStep(tape5.callback, ["3"])),
            ("d", SomeCheckpointStep(data_container=dc)
             ),
        ],
        [])

    tape6 = TapeCallbackFunction()
    tape_checkpoint_saved_inside_subpipeline_last_step = ResumablePipelineTestCase(
        tape6,
        data_inputs,
        expected_outputs,
        [
            ("a", TransformCallbackStep(tape6.callback, ["1"])),
            Pipeline([
                ("b", TransformCallbackStep(tape6.callback, ["2"])),
                ("d", SomeCheckpointStep(data_container=dc)
                 ),
            ]),
            ("e", TransformCallbackStep(tape6.callback, ["3"])),
            ("f", TransformCallbackStep(tape6.callback, ["4"])),
        ],
        ["3", "4"])

    tape7 = TapeCallbackFunction()
    tape_checkpoint_saved_inside_subpipeline_first_step = ResumablePipelineTestCase(
        tape7,
        data_inputs,
        expected_outputs,
        [
            ("a", TransformCallbackStep(tape7.callback, ["1"])),
            Pipeline([
                ("d", SomeCheckpointStep(data_container=dc)
                 ),
                ("b", TransformCallbackStep(tape7.callback, ["2"])),
            ]),
            ("e", TransformCallbackStep(tape7.callback, ["3"])),
            ("f", TransformCallbackStep(tape7.callback, ["4"])),
        ],
        ["2", "3", "4"])

    tape8 = TapeCallbackFunction()
    tape_checkpoint_saved_inside_subpipeline_step_in_the_middle = ResumablePipelineTestCase(
        tape8,
        data_inputs,
        expected_outputs,
        [
            ("a", TransformCallbackStep(tape8.callback, ["1"])),
            Pipeline([
                ("b", TransformCallbackStep(tape8.callback, ["2"])),
                ("d", SomeCheckpointStep(data_container=dc)
                 ),
                ("e", TransformCallbackStep(tape8.callback, ["3"])),
            ]),
            ("f", TransformCallbackStep(tape8.callback, ["4"])),
        ],
        ["3", "4"])

    tape9 = TapeCallbackFunction()
    tape_checkpoint_saved_inside_subpipeline_of_subpipeline = ResumablePipelineTestCase(
        tape9,
        data_inputs,
        expected_outputs,
        [
            ("a", TransformCallbackStep(tape9.callback, ["1"])),
            Pipeline([
                ("b", TransformCallbackStep(tape9.callback, ["2"])),
                Pipeline([
                    ("e", TransformCallbackStep(tape9.callback, ["3"])),
                    ("d", SomeCheckpointStep(data_container=dc)
                     ),
                    ("f", TransformCallbackStep(tape9.callback, ["4"])),
                ]),
                ("g", TransformCallbackStep(tape9.callback, ["5"])),
            ]),
            ("h", TransformCallbackStep(tape9.callback, ["6"])),
        ],
        ["4", "5", "6"])

    tape10 = TapeCallbackFunction()
    tape_saved_checkpoint_after_another_saved_checkpoint = ResumablePipelineTestCase(
        tape10,
        data_inputs,
        expected_outputs,
        [
            ("a", TransformCallbackStep(tape10.callback, ["1"])),
            ("b", SomeCheckpointStep(data_container=dc)),
            ("c", TransformCallbackStep(tape10.callback, ["2"])),
            ("b", SomeCheckpointStep(data_container=dc)),
            ("d", TransformCallbackStep(tape10.callback, ["3"]))
        ],
        ["3"])

    tape11 = TapeCallbackFunction()
    tape_multiple_checkpoint_in_a_row = ResumablePipelineTestCase(
        tape11,
        data_inputs,
        expected_outputs,
        [
            ("a", TransformCallbackStep(tape11.callback, ["1"])),
            ("pickle_1", SomeCheckpointStep(data_container=dc)),
            ("pickle_2", SomeCheckpointStep(data_container=dc)),
            ("c", TransformCallbackStep(tape11.callback, ["2"])),
            ("d", TransformCallbackStep(tape11.callback, ["3"]))
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
def test_should_fit_transform_each_steps(test_case: ResumablePipelineTestCase):
    pipeline = Pipeline(
        steps=test_case.steps
    )

    actual_pipeline, actual_data_inputs = pipeline.fit_transform(test_case.data_inputs, test_case.expected_outputs)

    actual_tape = test_case.tape.get_name_tape()
    assert isinstance(actual_pipeline, Pipeline)
    assert actual_tape == test_case.expected_tape
    assert np.array_equal(actual_data_inputs, test_case.data_inputs)


@pytest.mark.parametrize("test_case", create_test_cases())
def test_should_fit_each_steps(test_case: ResumablePipelineTestCase):
    pipeline = Pipeline(
        steps=test_case.steps
    )

    actual_pipeline = pipeline.fit(test_case.data_inputs, test_case.expected_outputs)

    actual_tape = test_case.tape.get_name_tape()
    assert isinstance(actual_pipeline, Pipeline)
    assert actual_tape == test_case.expected_tape


@pytest.mark.parametrize("test_case", create_test_cases())
def test_should_transform_each_steps(test_case: ResumablePipelineTestCase):
    pipeline = Pipeline(
        steps=test_case.steps
    )

    actual_data_inputs = pipeline.transform(test_case.data_inputs)

    actual_tape = test_case.tape.get_name_tape()
    assert actual_tape == test_case.expected_tape
    assert np.array_equal(actual_data_inputs, test_case.data_inputs)
