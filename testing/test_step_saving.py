import os

import numpy as np
from joblib import dump
from py._path.local import LocalPath

from neuraxle.base import BaseStep
from neuraxle.checkpoints import PickleCheckpointStep
from neuraxle.pipeline import ResumablePipeline

OUTPUT = "OUTPUT"
ROOT = 'ResumablePipeline'
CHECKPOINT = 'checkpoint'
SOME_STEP_2 = 'some_step2'
PIPELINE_2 = 'pipeline2'
SOME_STEP_1 = 'some_step1'
SOME_STEP_3 = 'some_step3'

EXPECTED_OUTPUTS = [0, 48, 96, 144, 192, 240, 288, 336, 384, 432]

class MultiplyBy(BaseStep):
    def __init__(self, multiply_by):
        super().__init__()
        self.multiply_by = multiply_by

    def transform(self, data_inputs):
        return data_inputs * self.multiply_by


def test_resumable_pipeline_fit_transform_should_save_all_pipeline_steps(tmpdir: LocalPath):
    p = ResumablePipeline([
        (SOME_STEP_1, MultiplyBy(multiply_by=2)),
        (PIPELINE_2, ResumablePipeline([
            (SOME_STEP_2, MultiplyBy(multiply_by=4)),
            (CHECKPOINT, PickleCheckpointStep(cache_folder=tmpdir)),
            (SOME_STEP_3, MultiplyBy(multiply_by=6)),
        ]))
    ], cache_folder=tmpdir)
    p.name = ROOT

    p, outputs = p.fit_transform(
        np.array(range(10)),
        np.array(range(10))
    )

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)
    assert os.path.exists(os.path.join(tmpdir, ROOT, '{0}.joblib'.format(ROOT)))
    assert os.path.exists(os.path.join(tmpdir, ROOT, SOME_STEP_1, '{0}.joblib'.format(SOME_STEP_1)))
    assert os.path.exists(
        os.path.join(tmpdir, ROOT, PIPELINE_2, '{0}.joblib'.format(PIPELINE_2)))
    assert os.path.exists(
        os.path.join(tmpdir, ROOT, PIPELINE_2, SOME_STEP_2, '{0}.joblib'.format(SOME_STEP_2)))
    assert os.path.exists(
        os.path.join(tmpdir, ROOT, PIPELINE_2, SOME_STEP_3, '{0}.joblib'.format(SOME_STEP_3)))



def test_resumable_pipeline_fit_transform_should_load_all_pipeline_steps(tmpdir: LocalPath):
    # Given
    os.makedirs(os.path.join(tmpdir, ROOT))
    os.path.join(tmpdir, ROOT, '{0}.joblib'.format(ROOT))

    os.makedirs(os.path.join(tmpdir, ROOT, SOME_STEP_1))
    dump(
        MultiplyBy(multiply_by=2),
        os.path.join(tmpdir, ROOT, SOME_STEP_1, '{0}.joblib'.format(SOME_STEP_1))
    )

    os.makedirs(os.path.join(tmpdir, ROOT, PIPELINE_2))
    os.path.join(tmpdir, ROOT, PIPELINE_2, '{0}.joblib'.format(PIPELINE_2))

    os.makedirs(os.path.join(tmpdir, ROOT, PIPELINE_2, SOME_STEP_2))
    dump(
        MultiplyBy(multiply_by=4),
        os.path.join(tmpdir, ROOT, PIPELINE_2, SOME_STEP_2, '{0}.joblib'.format(SOME_STEP_2))
    )

    os.makedirs(os.path.join(tmpdir, ROOT, PIPELINE_2, SOME_STEP_3))
    dump(
        MultiplyBy(multiply_by=6),
        os.path.join(tmpdir, ROOT, PIPELINE_2, SOME_STEP_3, '{0}.joblib'.format(SOME_STEP_3))
    )

    p = ResumablePipeline([
        (SOME_STEP_1, MultiplyBy(multiply_by=1)),
        (PIPELINE_2, ResumablePipeline([
            (SOME_STEP_2, MultiplyBy(multiply_by=1)),
            (CHECKPOINT, PickleCheckpointStep(cache_folder=tmpdir)),
            (SOME_STEP_3, MultiplyBy(multiply_by=1)),
        ]))
    ], cache_folder=tmpdir)
    p.name = ROOT

    # When
    p, outputs = p.fit_transform(
        np.array(range(10)),
        np.array(range(10))
    )

    # Then
    assert np.array_equal(outputs, EXPECTED_OUTPUTS)
