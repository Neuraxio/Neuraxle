"""
Tests for pipeline hashing
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

from typing import List, Any

import numpy as np
import pytest

from neuraxle.base import NamedTupleList, BaseHasher
from neuraxle.checkpoints import BaseCheckpointStep
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import TransformCallbackStep, TapeCallbackFunction
from testing.test_resumable_pipeline import SomeCheckpointStep


class ResumablePipelineTestCase:
    def __init__(self, tape, steps: NamedTupleList, expected_tape: List[str], di, eo=None, expected_rehashed_ids=None):
        self.tape = tape
        self.expected_outputs = eo
        self.data_inputs = di
        self.steps = steps
        self.expected_tape = expected_tape
        self.expected_rehashed_ids = expected_rehashed_ids


class MockHasher(BaseHasher):
    def hash(self, current_ids, hyperparameters: HyperparameterSamples, data_inputs: Any):
        if current_ids is None:
            current_ids = list(range(len(data_inputs)))
            current_ids = [str(c) for c in current_ids]

        if len(hyperparameters) == 0:
            return current_ids
        else:
            items = ",".join([str(value) for prop, value in hyperparameters.to_flat().items()])

        return [
            ",".join([str(current_id), items])
            for current_id in current_ids
        ]


def create_callback_step(tape_step_name, hyperparams):
    step = (
        tape_step_name,
        TransformCallbackStep(
            callback_function=TapeCallbackFunction().callback,
            more_arguments=[tape_step_name],
            hyperparams=HyperparameterSamples(hyperparams)
        )
    )
    return step


def find_checkpoint(steps):
    for name, step in steps:
        if isinstance(step, Pipeline):
            return find_checkpoint(step.steps_as_tuple)
        if isinstance(step, BaseCheckpointStep):
            return step
    return None


def set_hasher(steps, hasher):
    for name, step in steps:
        step.set_hasher(hasher)
        if isinstance(step, Pipeline):
            set_hasher(step.steps_as_tuple, hasher)
        if isinstance(step, BaseCheckpointStep):
            return step
    return None


def create_test_cases():
    one_step_with_empty_hyperparameters = ResumablePipelineTestCase(
        tape=TapeCallbackFunction(),
        steps=[
            create_callback_step("a", {}),
            ("checkpoint", SomeCheckpointStep()),
        ],
        di=np.array([1, 2]),
        eo=np.array([1, 2]),
        expected_tape=["a"],
        expected_rehashed_ids=['0', '1'],
    )

    steps_with_empty_hyperparameters = ResumablePipelineTestCase(
        tape=TapeCallbackFunction(),
        steps=[
            create_callback_step("a", {}),
            create_callback_step("b", {}),
            ("checkpoint", SomeCheckpointStep()),
        ],
        di=np.array([1, 2]),
        eo=np.array([1, 2]),
        expected_tape=["a", "b"],
        expected_rehashed_ids=['0', '1'],
    )

    steps_with_empty_hyperparameters_sub_pipeline = ResumablePipelineTestCase(
        tape=TapeCallbackFunction(),
        steps=[
            create_callback_step("a", {}),
            create_callback_step("b", {}),
            Pipeline([
                create_callback_step("c", {}),
                create_callback_step("d", {}),
                ("checkpoint", SomeCheckpointStep()),
            ])
        ],
        di=np.array([1, 2]),
        eo=np.array([1, 2]),
        expected_tape=["a", "b"],
        expected_rehashed_ids=['0', '1'],
    )

    one_step_with_hyperparameters = ResumablePipelineTestCase(
        tape=TapeCallbackFunction(),
        steps=[
            create_callback_step("a", {"a__learning_rate": 1}),
            ("checkpoint", SomeCheckpointStep()),
        ],
        di=np.array([1, 2]),
        eo=np.array([1, 2]),
        expected_tape=["a"],
        expected_rehashed_ids=['0,1', '1,1'],
    )

    steps_with_hyperparameters = ResumablePipelineTestCase(
        tape=TapeCallbackFunction(),
        steps=[
            create_callback_step("a", {"a__learning_rate": 1}),
            create_callback_step("b", {"b__learning_rate": 2}),
            ("checkpoint", SomeCheckpointStep()),
        ],
        di=np.array([1, 2]),
        eo=np.array([1, 2]),
        expected_tape=["a", "b"],
        expected_rehashed_ids=['0,1,2', '1,1,2'],
    )

    steps_with_hyperparameters_sub_pipeline = ResumablePipelineTestCase(
        tape=TapeCallbackFunction(),
        steps=[
            create_callback_step("a", {"a__learning_rate": 1}),
            create_callback_step("b", {"b__learning_rate": 2}),
            Pipeline([
                create_callback_step("c", {"c__learning_rate": 3}),
                create_callback_step("d", {"d__learning_rate": 4}),
                ("checkpoint", SomeCheckpointStep()),
            ])
        ],
        di=np.array([1, 2]),
        eo=np.array([1, 2]),
        expected_tape=["a", "b"],
        expected_rehashed_ids=['0,1,2,3,4', '1,1,2,3,4'],
    )

    one_step_with_nested_hyperparameters = ResumablePipelineTestCase(
        tape=TapeCallbackFunction(),
        steps=[
            create_callback_step("a", {"a": {"learning_rate": 1}}),
            ("checkpoint", SomeCheckpointStep()),
        ],
        di=np.array([1, 2]),
        eo=np.array([1, 2]),
        expected_tape=["a"],
        expected_rehashed_ids=['0,1', '1,1'],
    )

    steps_with_nested_hyperparameters = ResumablePipelineTestCase(
        tape=TapeCallbackFunction(),
        steps=[
            create_callback_step("a", {"a": {"learning_rate": 1}}),
            create_callback_step("b", {"b": {"learning_rate": 2}}),
            ("checkpoint", SomeCheckpointStep()),
        ],
        di=np.array([1, 2]),
        eo=np.array([1, 2]),
        expected_tape=["a", "b"],
        expected_rehashed_ids=['0,1,2', '1,1,2'],
    )

    steps_with_nested_hyperparameters_sub_pipeline = ResumablePipelineTestCase(
        tape=TapeCallbackFunction(),
        steps=[
            create_callback_step("a", {"a": {"learning_rate": 1}}),
            create_callback_step("b", {"b": {"learning_rate": 2}}),
            Pipeline([
                create_callback_step("c", {"c": {"learning_rate": 3}}),
                create_callback_step("d", {"d": {"learning_rate": 4}}),
                ("checkpoint", SomeCheckpointStep()),
            ])
        ],
        di=np.array([1, 2]),
        eo=np.array([1, 2]),
        expected_tape=["a", "b"],
        expected_rehashed_ids=['0,1,2,3,4', '1,1,2,3,4'],
    )


    return [
        one_step_with_empty_hyperparameters,
        steps_with_empty_hyperparameters,
        steps_with_empty_hyperparameters_sub_pipeline,
        one_step_with_hyperparameters,
        steps_with_hyperparameters,
        steps_with_hyperparameters_sub_pipeline,
        one_step_with_nested_hyperparameters,
        steps_with_nested_hyperparameters,
        steps_with_nested_hyperparameters_sub_pipeline
    ]


@pytest.mark.parametrize("test_case", create_test_cases())
def test_transform_should_rehash_hyperparameters_for_each_steps(test_case: ResumablePipelineTestCase):
    pipeline = Pipeline(steps=test_case.steps)
    pipeline.set_hasher(MockHasher())
    set_hasher(pipeline.steps_as_tuple, MockHasher())

    pipeline.transform(test_case.data_inputs)

    mocked_checkpoint = find_checkpoint(pipeline.steps_as_tuple)
    actual_rehashed_ids = mocked_checkpoint.saved_data_container.current_ids
    assert actual_rehashed_ids == test_case.expected_rehashed_ids


@pytest.mark.parametrize("test_case", create_test_cases())
def test_fit_should_rehash_hyperparameters_for_each_steps(test_case: ResumablePipelineTestCase):
    pipeline = Pipeline(steps=test_case.steps)
    pipeline.set_hasher(MockHasher())
    set_hasher(pipeline.steps_as_tuple, MockHasher())

    pipeline.fit(test_case.data_inputs, test_case.expected_outputs)

    mocked_checkpoint = find_checkpoint(pipeline.steps_as_tuple)
    actual_rehashed_ids = mocked_checkpoint.saved_data_container.current_ids
    assert actual_rehashed_ids == test_case.expected_rehashed_ids


@pytest.mark.parametrize("test_case", create_test_cases())
def test_fit_transform_should_rehash_hyperparameters_for_each_steps(test_case: ResumablePipelineTestCase):
    pipeline = Pipeline(steps=test_case.steps)
    pipeline.set_hasher(MockHasher())
    set_hasher(pipeline.steps_as_tuple, MockHasher())

    pipeline.fit_transform(test_case.data_inputs, test_case.expected_outputs)

    mocked_checkpoint = find_checkpoint(pipeline.steps_as_tuple)
    actual_rehashed_ids = mocked_checkpoint.saved_data_container.current_ids
    assert actual_rehashed_ids == test_case.expected_rehashed_ids
