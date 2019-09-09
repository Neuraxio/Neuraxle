import string
from typing import List

import numpy as np
import pytest

from neuraxle.base import BaseStep, DataContainer
from neuraxle.checkpoints import BaseCheckpointStep
from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import TransformCallbackStep, TapeCallbackFunction


class SomeCheckpointStep(BaseCheckpointStep):
    def __init__(self, data_container: DataContainer = None):
        super().__init__()
        self.data_container = data_container
        self.saved = False
        self.saved_data_container = None

    def set_checkpoint_path(self, path):
        pass

    def read_checkpoint(self, data_container: DataContainer):
        return self.data_container

    def save_checkpoint(self, data_container: DataContainer):
        self.saved_data_container = data_container
        self.saved = True

    def should_resume(self, data_container) -> bool:
        return self.data_container is not None


data_inputs = np.ones((1, 1))
expected_outputs = np.ones((1, 1))
dc = DataContainer(ids=range(len(data_inputs)), data_inputs=data_inputs, expected_outputs=expected_outputs)
chekpoint = SomeCheckpointStep(dc)
chekpoint_not_saved = SomeCheckpointStep(None)
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
        ("b", SomeCheckpointStep(data_container=None)
         ),
        ("c", TransformCallbackStep(tape.callback, ["2"])),
        ("d", TransformCallbackStep(tape.callback, ["3"]))
    ],
    ["1", "2", "3"])

tape_checkpoint_saved_after_first_step_test_arguments = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        ("b", SomeCheckpointStep(data_container=dc)
         ),
        ("c", TransformCallbackStep(tape.callback, ["2"])),
        ("d", TransformCallbackStep(tape.callback, ["3"]))
    ],
    ["2", "3"])

tape_checkpoint_saved_after_second_step_test_arguments = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        ("b", TransformCallbackStep(tape.callback, ["2"])),
        ("c", SomeCheckpointStep(data_container=dc)
         ),
        ("d", TransformCallbackStep(tape.callback, ["3"]))
    ],
    ["3"])

tape_checkpoint_saved_after_last_step_test_arguments = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        ("b", TransformCallbackStep(tape.callback, ["2"])),
        ("c", TransformCallbackStep(tape.callback, ["3"])),
        ("d", SomeCheckpointStep(data_container=dc)
         ),
    ],
    [])

tape_checkpoint_saved_inside_subpipeline_last_step = (
    [
        ("a", TransformCallbackStep(tape.callback, ["1"])),
        Pipeline([
            ("b", TransformCallbackStep(tape.callback, ["2"])),
            ("d", SomeCheckpointStep(data_container=dc)
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
            ("d", SomeCheckpointStep(data_container=dc)
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
            ("d", SomeCheckpointStep(data_container=dc)
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
                ("d", SomeCheckpointStep(data_container=dc)
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
