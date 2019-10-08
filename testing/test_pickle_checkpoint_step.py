"""
Tests for pickle checkpoint steps
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

import os
import pickle

import numpy as np
from py._path.local import LocalPath

from neuraxle.base import DataContainer, ExecutionContext
from neuraxle.base import NonFittableMixin
from neuraxle.checkpoints import PickleCheckpointStep
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.pipeline import ResumablePipeline
from neuraxle.steps.util import TapeCallbackFunction, TransformCallbackStep, BaseCallbackStep
from testing.steps.test_output_transformer_wrapper import MultiplyBy2OutputTransformer

EXPECTED_TAPE_AFTER_CHECKPOINT = ["2", "3"]

data_inputs = np.array([1, 2])
expected_outputs = np.array([2, 3])
expected_rehashed_data_inputs = ['44f9d6dd8b6ccae571ca04525c3eaffa', '898a67b2f5eeae6393ca4b3162ba8e3d']


class DifferentCallbackStep(NonFittableMixin, BaseCallbackStep):
    def transform(self, data_inputs):
        self._callback(data_inputs)
        return data_inputs

    def transform_one(self, data_input):
        self._callback(data_input)
        return data_input


def create_pipeline(tmpdir, pickle_checkpoint_step, tape, hyperparameters=None, different=False, save_pipeline=True):
    if different:
        pipeline = ResumablePipeline(
            steps=[
                ('a',
                 DifferentCallbackStep(tape.callback, ["1"], hyperparams=hyperparameters)),
                ('pickle_checkpoint', pickle_checkpoint_step),
                ('c', TransformCallbackStep(tape.callback, ["2"])),
                ('d', TransformCallbackStep(tape.callback, ["3"]))
            ]
        )
    else:
        pipeline = ResumablePipeline(
            steps=[
                ('a',
                 TransformCallbackStep(tape.callback, ["1"], hyperparams=hyperparameters)),
                ('pickle_checkpoint', pickle_checkpoint_step),
                ('c', TransformCallbackStep(tape.callback, ["2"])),
                ('d', TransformCallbackStep(tape.callback, ["3"]))
            ]
        )
    return pipeline


def test_when_no_hyperparams_should_save_checkpoint_pickle(tmpdir: LocalPath):
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)
    pipeline = create_pipeline(tmpdir, pickle_checkpoint_step, tape)

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == ["1", "2", "3"]
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(0))
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(1))


def test_when_hyperparams_should_save_checkpoint_pickle(tmpdir: LocalPath):
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)
    pipeline = create_pipeline(tmpdir, pickle_checkpoint_step, tape, HyperparameterSamples({"a__learning_rate": 1}))

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == ["1", "2", "3"]
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(expected_rehashed_data_inputs[0]))
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(expected_rehashed_data_inputs[1]))


def test_when_no_hyperparams_and_saved_same_pipeline_should_load_checkpoint_pickle(tmpdir: LocalPath):
    # Given
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)

    # When
    pipeline_save = create_pipeline(
        tmpdir=tmpdir,
        pickle_checkpoint_step=create_pickle_checkpoint_step(tmpdir),
        tape=TapeCallbackFunction()
    )
    pipeline_save.fit_transform(data_inputs, expected_outputs)

    pipeline_load = create_pipeline(
        tmpdir=tmpdir,
        pickle_checkpoint_step=pickle_checkpoint_step,
        tape=tape
    )
    pipeline_load, actual_data_inputs = pipeline_load.fit_transform(data_inputs, expected_outputs)

    # Then
    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == EXPECTED_TAPE_AFTER_CHECKPOINT


def test_when_hyperparams_and_saved_same_pipeline_should_load_checkpoint_pickle(tmpdir: LocalPath):
    # Given
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)

    # When
    pipeline_save = create_pipeline(
        tmpdir=tmpdir,
        pickle_checkpoint_step=create_pickle_checkpoint_step(tmpdir),
        tape=TapeCallbackFunction(),
        hyperparameters=HyperparameterSamples({"a__learning_rate": 1})
    )
    pipeline_save.fit_transform(data_inputs, expected_outputs)

    pipeline_load = create_pipeline(
        tmpdir=tmpdir,
        pickle_checkpoint_step=pickle_checkpoint_step,
        tape=tape,
        hyperparameters=HyperparameterSamples({"a__learning_rate": 1})
    )
    pipeline_load, actual_data_inputs = pipeline_load.fit_transform(data_inputs, expected_outputs)

    # Then
    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == EXPECTED_TAPE_AFTER_CHECKPOINT


def test_when_hyperparams_and_saved_different_pipeline_should_not_load_checkpoint_pickle(tmpdir: LocalPath):
    # Given
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)

    # When
    pipeline_save = create_pipeline(
        tmpdir=tmpdir,
        pickle_checkpoint_step=create_pickle_checkpoint_step(tmpdir),
        tape=TapeCallbackFunction(),
        hyperparameters=HyperparameterSamples({"a__learning_rate": 1}),
        different=True
    )
    pipeline_save.fit_transform(data_inputs, expected_outputs)

    pipeline_load = create_pipeline(
        tmpdir=tmpdir,
        pickle_checkpoint_step=pickle_checkpoint_step,
        tape=tape,
        hyperparameters=HyperparameterSamples({"a__learning_rate": 1})
    )
    pipeline_load, actual_data_inputs = pipeline_load.fit_transform(data_inputs, expected_outputs)

    # Then
    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == ["1", "2", "3"]


def test_when_hyperparams_and_saved_no_pipeline_should_not_load_checkpoint_pickle(tmpdir: LocalPath):
    # Given
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)

    # When
    pipeline_save = create_pipeline(
        tmpdir=tmpdir,
        pickle_checkpoint_step=create_pickle_checkpoint_step(tmpdir),
        tape=TapeCallbackFunction(),
        hyperparameters=HyperparameterSamples({"a__learning_rate": 1}),
        different=True,
        save_pipeline=False
    )
    pipeline_save.fit_transform(data_inputs, expected_outputs)

    pipeline_load = create_pipeline(
        tmpdir=tmpdir,
        pickle_checkpoint_step=pickle_checkpoint_step,
        tape=tape,
        hyperparameters=HyperparameterSamples({"a__learning_rate": 1})
    )
    pipeline_load, actual_data_inputs = pipeline_load.fit_transform(data_inputs, expected_outputs)

    # Then
    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == ["1", "2", "3"]


def setup_pickle_checkpoint(current_id, data_input, expected_output, pickle_checkpoint_step):
    with open(pickle_checkpoint_step.get_checkpoint_file_path(current_id), 'wb') as file:
        pickle.dump((current_id, data_input, expected_output), file)


def create_pickle_checkpoint_step(tmpdir):
    pickle_checkpoint_step = PickleCheckpointStep(cache_folder=tmpdir)
    pickle_checkpoint_step.set_checkpoint_path(os.path.join('Pipeline', 'pickle_checkpoint'))

    return pickle_checkpoint_step


def test_pickle_checkpoint_step_should_load_data_container(tmpdir: LocalPath):
    initial_data_inputs = [1, 2]
    initial_expected_outputs = [2, 3]

    create_pipeline_output_transformer = lambda: Pipeline(
        [
            ('output_transformer', MultiplyBy2OutputTransformer()),
            ('pickle_checkpoint', create_pickle_checkpoint_step(tmpdir)),
            ('output_transformer', MultiplyBy2OutputTransformer()),
        ]
    )

    create_pipeline_output_transformer().fit_transform(
        data_inputs=initial_data_inputs, expected_outputs=initial_expected_outputs
    )
    transformer = create_pipeline_output_transformer()
    actual_data_container = transformer.handle_transform(
        DataContainer(current_ids=[0, 1], data_inputs=initial_data_inputs, expected_outputs=initial_expected_outputs),
        ExecutionContext.from_root(transformer)
    )

    assert np.array_equal(actual_data_container.data_inputs, [4, 8])
    assert np.array_equal(actual_data_container.expected_outputs, [8, 12])
