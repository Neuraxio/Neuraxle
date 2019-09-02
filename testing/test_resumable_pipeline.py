import string
from typing import List

import numpy as np
import pytest

from neuraxle.base import BaseStep
from neuraxle.checkpoints import BaseCheckpointStep
from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import TransformCallbackStep, TapeCallbackFunction


class SomeCheckpointStep(BaseCheckpointStep):
    def __init__(self, checkpoint_data_inputs, checkpoint_expected_outputs):
        super().__init__()
        self.checkpoint_data_inputs = checkpoint_data_inputs
        self.checkpoint_expected_outputs = checkpoint_expected_outputs
        self.saved = False
        self.data_inputs_checkpoint = None
        self.expected_outputs_checkpoint = None

    def set_checkpoint_path(self, path):
        pass

    def read_checkpoint(self):
        return self.checkpoint_data_inputs, self.checkpoint_expected_outputs

    def save_checkpoint(self, data_inputs, expected_outputs=None):
        self.data_inputs_checkpoint = data_inputs
        self.expected_outputs_checkpoint = expected_outputs
        self.saved = True


data_inputs = np.ones((1, 1))
expected_outputs = np.ones((1, 1))
chekpoint = SomeCheckpointStep(data_inputs, expected_outputs)
chekpoint_not_saved = SomeCheckpointStep(None, None)
tape = TapeCallbackFunction()

tape_without_checkpoint_test_arguments = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        ("b", TransformCallbackStep(tape.callback, ["2"])),
        ("c", TransformCallbackStep(tape.callback, ["3"]))
    ],
    ["1", "2", "3"])

tape_checkpoint_not_saved_test_arguments = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        ("b", SomeCheckpointStep(
            checkpoint_data_inputs=None,
            checkpoint_expected_outputs=None)
         ),
        ("c", TransformCallbackStep(tape.callback, ["2"])),
        ("d", TransformCallbackStep(tape.callback, ["3"]))
    ],
    ["1", "2", "3"])

tape_checkpoint_saved_after_first_step_test_arguments = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        ("b", SomeCheckpointStep(
            checkpoint_data_inputs=data_inputs,
            checkpoint_expected_outputs=expected_outputs)
         ),
        ("c", TransformCallbackStep(tape.callback, ["2"])),
        ("d", TransformCallbackStep(tape.callback, ["3"]))
    ],
    ["2", "3"])

tape_checkpoint_saved_after_second_step_test_arguments = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        ("b", TransformCallbackStep(tape.callback, ["2"])),
        ("c", SomeCheckpointStep(
            checkpoint_data_inputs=data_inputs,
            checkpoint_expected_outputs=expected_outputs)
         ),
        ("d", TransformCallbackStep(tape.callback, ["3"]))
    ],
    ["3"])

tape_checkpoint_saved_after_last_step_test_arguments = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        ("b", TransformCallbackStep(tape.callback, ["2"])),
        ("c", TransformCallbackStep(tape.callback, ["3"])),
        ("d", SomeCheckpointStep(
            checkpoint_data_inputs=data_inputs,
            checkpoint_expected_outputs=expected_outputs)
         ),
    ],
    [])

tape_checkpoint_saved_inside_subpipeline_last_step = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        Pipeline([
            ("b", TransformCallbackStep(tape.callback, ["2"])),
            ("d", SomeCheckpointStep(
                checkpoint_data_inputs=data_inputs,
                checkpoint_expected_outputs=expected_outputs)
             ),
        ]),
        ("e", TransformCallbackStep(tape.callback, ["3"])),
        ("f", TransformCallbackStep(tape.callback, ["4"])),
    ],
    ["3", "4"])

tape_checkpoint_saved_inside_subpipeline_first_step = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        Pipeline([
            ("d", SomeCheckpointStep(
                checkpoint_data_inputs=data_inputs,
                checkpoint_expected_outputs=expected_outputs)
             ),
            ("b", TransformCallbackStep(tape.callback, ["2"])),
        ]),
        ("e", TransformCallbackStep(tape.callback, ["3"])),
        ("f", TransformCallbackStep(tape.callback, ["4"])),
    ],
    ["2", "3", "4"])

tape_checkpoint_saved_inside_subpipeline_step_in_the_middle = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        Pipeline([
            ("b", TransformCallbackStep(tape.callback, ["2"])),
            ("d", SomeCheckpointStep(
                checkpoint_data_inputs=data_inputs,
                checkpoint_expected_outputs=expected_outputs)
             ),
            ("e", TransformCallbackStep(tape.callback, ["3"])),
        ]),
        ("f", TransformCallbackStep(tape.callback, ["4"])),
    ],
    ["3", "4"])

tape_checkpoint_saved_inside_subpipeline_of_subpipeline = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        Pipeline([
            ("b", TransformCallbackStep(tape.callback, ["2"])),
            Pipeline([
                ("e", TransformCallbackStep(tape.callback, ["3"])),
                ("d", SomeCheckpointStep(
                    checkpoint_data_inputs=data_inputs,
                    checkpoint_expected_outputs=expected_outputs)
                 ),
                ("f", TransformCallbackStep(tape.callback, ["4"])),
            ]),
            ("g", TransformCallbackStep(tape.callback, ["5"])),
        ]),
        ("h", TransformCallbackStep(tape.callback, ["6"])),
    ],
    ["4", "5", "6"])


@pytest.mark.parametrize("steps,expected_tape", [
    tape_without_checkpoint_test_arguments,
    tape_checkpoint_not_saved_test_arguments,
    tape_checkpoint_saved_after_first_step_test_arguments,
    tape_checkpoint_saved_after_second_step_test_arguments,
    tape_checkpoint_saved_after_last_step_test_arguments,
    tape_checkpoint_saved_inside_subpipeline_first_step,
    tape_checkpoint_saved_inside_subpipeline_last_step,
    tape_checkpoint_saved_inside_subpipeline_step_in_the_middle,
    tape_checkpoint_saved_inside_subpipeline_of_subpipeline,
])
def test_fit_transform(steps: List[BaseStep], expected_tape: List[str]):
    tape.data = []
    tape.name_tape = []
    pipeline = Pipeline(
        steps=steps
    )

    actual_pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert isinstance(actual_pipeline, Pipeline)
    assert actual_tape == expected_tape
    assert np.array_equal(actual_data_inputs, data_inputs)


@pytest.mark.parametrize("steps,expected_tape", [
    tape_without_checkpoint_test_arguments,
    tape_checkpoint_not_saved_test_arguments,
    tape_checkpoint_saved_after_first_step_test_arguments,
    tape_checkpoint_saved_after_second_step_test_arguments,
    tape_checkpoint_saved_after_last_step_test_arguments,
    tape_checkpoint_saved_inside_subpipeline_first_step,
    tape_checkpoint_saved_inside_subpipeline_last_step,
    tape_checkpoint_saved_inside_subpipeline_step_in_the_middle,
    tape_checkpoint_saved_inside_subpipeline_of_subpipeline,
])
def test_should_fit_each_steps(steps: List[BaseStep], expected_tape: List[str]):
    tape.data = []
    tape.name_tape = []
    pipeline = Pipeline(
        steps=steps
    )

    actual_pipeline = pipeline.fit(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert isinstance(actual_pipeline, Pipeline)
    assert actual_tape == expected_tape


@pytest.mark.parametrize("steps,expected_tape", [
    tape_without_checkpoint_test_arguments,
    tape_checkpoint_not_saved_test_arguments,
    tape_checkpoint_saved_after_first_step_test_arguments,
    tape_checkpoint_saved_after_second_step_test_arguments,
    tape_checkpoint_saved_after_last_step_test_arguments,
    tape_checkpoint_saved_inside_subpipeline_first_step,
    tape_checkpoint_saved_inside_subpipeline_last_step,
    tape_checkpoint_saved_inside_subpipeline_step_in_the_middle,
    tape_checkpoint_saved_inside_subpipeline_of_subpipeline,
])
def test_should_transform_each_steps(steps: List[BaseStep], expected_tape: List[str]):
    pipeline = Pipeline(
        steps=steps
    )
    pipeline = pipeline.fit(data_inputs)
    tape.data = []
    tape.name_tape = []

    actual_data_inputs = pipeline.transform(data_inputs)

    actual_tape = tape.get_name_tape()
    assert actual_tape == expected_tape
    assert np.array_equal(actual_data_inputs, data_inputs)
