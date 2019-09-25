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

from neuraxle.checkpoints import PickleCheckpointStep
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import TapeCallbackFunction, TransformCallbackStep

EXPECTED_TAPE_AFTER_CHECKPOINT = ["2", "3"]

data_inputs = np.array([1, 2])
expected_outputs = np.array([2, 3])
expected_rehashed_data_inputs = ['44f9d6dd8b6ccae571ca04525c3eaffa', '898a67b2f5eeae6393ca4b3162ba8e3d']


def create_pipeline(pickle_checkpoint_step, tape, hyperparameters = None):
    pipeline = Pipeline(
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
    pipeline = create_pipeline(pickle_checkpoint_step, tape)

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == ["1", "2", "3"]
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(0))
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(1))


def test_when_hyperparams_should_save_checkpoint_pickle(tmpdir: LocalPath):
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)
    pipeline = create_pipeline(pickle_checkpoint_step, tape, HyperparameterSamples({"a__learning_rate": 1}))

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == ["1", "2", "3"]
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(expected_rehashed_data_inputs[0]))
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(expected_rehashed_data_inputs[1]))


def test_when_no_hyperparams_should_load_checkpoint_pickle(tmpdir: LocalPath):
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)
    setup_pickle_checkpoint(
        current_id=0,
        data_input=data_inputs[0],
        expected_output=expected_outputs[0],
        pickle_checkpoint_step=pickle_checkpoint_step
    )
    setup_pickle_checkpoint(
        current_id=1,
        data_input=data_inputs[1],
        expected_output=expected_outputs[1],
        pickle_checkpoint_step=pickle_checkpoint_step
    )
    pipeline = create_pipeline(pickle_checkpoint_step, tape)

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == EXPECTED_TAPE_AFTER_CHECKPOINT


def test_when_hyperparams_should_load_checkpoint_pickle(tmpdir: LocalPath):
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = create_pickle_checkpoint_step(tmpdir)
    setup_pickle_checkpoint(
        current_id=expected_rehashed_data_inputs[0],
        data_input=data_inputs[0],
        expected_output=expected_outputs[0],
        pickle_checkpoint_step=pickle_checkpoint_step
    )
    setup_pickle_checkpoint(
        current_id=expected_rehashed_data_inputs[1],
        data_input=data_inputs[1],
        expected_output=expected_outputs[1],
        pickle_checkpoint_step=pickle_checkpoint_step
    )

    pipeline = create_pipeline(pickle_checkpoint_step, tape, HyperparameterSamples({"a__learning_rate": 1}))

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert np.array_equal(actual_data_inputs, data_inputs)
    assert actual_tape == EXPECTED_TAPE_AFTER_CHECKPOINT


def setup_pickle_checkpoint(current_id, data_input, expected_output, pickle_checkpoint_step):
    with open(pickle_checkpoint_step.get_checkpoint_file_path(current_id), 'wb') as file:
        pickle.dump((current_id, data_input, expected_output), file)


def create_pickle_checkpoint_step(tmpdir):
    pickle_checkpoint_step = PickleCheckpointStep(cache_folder=tmpdir)
    pickle_checkpoint_step.set_checkpoint_path(os.path.join('Pipeline', 'pickle_checkpoint'))
    return pickle_checkpoint_step
