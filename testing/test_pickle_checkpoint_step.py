import os
import pickle

import numpy as np
from py._path.local import LocalPath

from neuraxle.checkpoints import PickleCheckpointStep
from neuraxle.pipeline import Pipeline
from neuraxle.steps.util import TapeCallbackFunction, TransformCallbackStep

data_inputs = np.ones((1, 1))
expected_outputs = np.ones((2, 2))


def test_should_save_checkpoint_pickle(tmpdir: LocalPath):
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = PickleCheckpointStep('1', tmpdir)
    pipeline = Pipeline(
        steps=[
            TransformCallbackStep(tape.callback, ["1"]),
            pickle_checkpoint_step,
            TransformCallbackStep(tape.callback, ["2"]),
            TransformCallbackStep(tape.callback, ["3"])
        ]
    )

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert actual_data_inputs == data_inputs
    assert actual_tape == ["1", "2", "3"]
    assert os.path.exists(pickle_checkpoint_step.get_checkpoint_file_path(data_inputs))


def test_should_load_checkpoint_pickle(tmpdir: LocalPath):
    tape = TapeCallbackFunction()
    force_checkpoint_name = 'checkpoint_a'
    pickle_checkpoint_step = PickleCheckpointStep(
        force_checkpoint_name=force_checkpoint_name,
        cache_folder=tmpdir
    )
    pickle_checkpoint_step.set_checkpoint_path(force_checkpoint_name)
    with open(pickle_checkpoint_step.get_checkpoint_file_path(data_inputs), 'wb') as file:
        pickle.dump(data_inputs, file)

    pipeline = Pipeline(
        steps=[
            ('a', TransformCallbackStep(tape.callback, ["1"])),
            ('b', TransformCallbackStep(tape.callback, ["2"])),
            (force_checkpoint_name, pickle_checkpoint_step),
            ('c', TransformCallbackStep(tape.callback, ["3"]))
        ]
    )

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert actual_data_inputs == data_inputs
    assert actual_tape == ["3"]
