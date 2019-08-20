import os
import pickle

import numpy as np

from neuraxle.checkpoints import PickleCheckpointStep
from neuraxle.pipeline import Pipeline
from neuraxle.runners import CheckpointPipelineRunner
from neuraxle.steps.util import TapeCallbackFunction, TransformCallbackStep

data_inputs = np.ones((1, 1))
expected_outputs = np.ones((2, 2))


def test_should_save_checkpoint_pickle(tmpdir: str):
    tape = TapeCallbackFunction()
    pickle_checkpoint_step = PickleCheckpointStep('1', tmpdir)
    pipeline = Pipeline(
        steps=[
            TransformCallbackStep(tape.callback, ["1"]),
            pickle_checkpoint_step,
            TransformCallbackStep(tape.callback, ["2"]),
            TransformCallbackStep(tape.callback, ["3"])
        ],
        pipeline_runner=CheckpointPipelineRunner()
    )

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert actual_data_inputs == data_inputs
    assert actual_tape == ["1", "2", "3"]
    assert os.path.exists(pickle_checkpoint_step.get_data_inputs_checkpoint_file_name())
    assert os.path.exists(pickle_checkpoint_step.get_expected_ouputs_file_name())


def test_should_load_checkpoint_pickle(tmpdir: str):
    tape = TapeCallbackFunction()
    force_checkpoint_name = 'checkpoint_a'
    pickle_checkpoint_step = PickleCheckpointStep(
        force_checkpoint_name=force_checkpoint_name,
        checkpoint_folder=tmpdir
    )
    with open(pickle_checkpoint_step.get_data_inputs_checkpoint_file_name(), 'wb') as file:
        pickle.dump(data_inputs, file)
    with open(pickle_checkpoint_step.get_expected_ouputs_file_name(), 'wb') as file:
        pickle.dump(expected_outputs, file)
    pipeline = Pipeline(
        steps=[
            ('a', TransformCallbackStep(tape.callback, ["1"])),
            ('b', TransformCallbackStep(tape.callback, ["2"])),
            (force_checkpoint_name, pickle_checkpoint_step),
            ('c', TransformCallbackStep(tape.callback, ["3"]))
        ],
        pipeline_runner=CheckpointPipelineRunner()
    )

    pipeline, actual_data_inputs = pipeline.fit_transform(data_inputs, expected_outputs)

    actual_tape = tape.get_name_tape()
    assert actual_data_inputs == data_inputs
    assert actual_tape == ["3"]
