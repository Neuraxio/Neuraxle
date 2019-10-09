import os

import numpy as np
from joblib import dump
from py._path.local import LocalPath

from neuraxle.base import BaseStep, TruncableJoblibStepSaver
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


def create_some_step3_path(tmpdir, create_dir=False):
    p = os.path.join(tmpdir, ROOT, PIPELINE_2, SOME_STEP_3, '{0}.joblib'.format(SOME_STEP_3))
    if create_dir:
        os.makedirs(os.path.join(tmpdir, ROOT, PIPELINE_2, SOME_STEP_3))
    return p


def create_some_step2_path(tmpdir, create_dir=False):
    p = os.path.join(tmpdir, ROOT, PIPELINE_2, SOME_STEP_2, '{0}.joblib'.format(SOME_STEP_2))
    if create_dir:
       os.makedirs(os.path.join(tmpdir, ROOT, PIPELINE_2, SOME_STEP_2))
    return p


def create_pipeline2_path(tmpdir, create_dir=False):
    p = os.path.join(tmpdir, ROOT, PIPELINE_2, '{0}.joblib'.format(PIPELINE_2))
    if create_dir:
        os.makedirs(os.path.join(tmpdir, ROOT, PIPELINE_2))
    return p


def create_some_step1_path(tmpdir, create_dir=False):
    p = os.path.join(tmpdir, ROOT, SOME_STEP_1, '{0}.joblib'.format(SOME_STEP_1))
    if create_dir:
        os.makedirs(os.path.join(tmpdir, ROOT, SOME_STEP_1))
    return p


def create_root_path(tmpdir, create_dir=False):
    p = os.path.join(tmpdir, ROOT, '{0}.joblib'.format(ROOT))
    if create_dir:
        os.makedirs(os.path.join(tmpdir, ROOT))
    return p


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
    assert os.path.exists(create_root_path(tmpdir))
    assert os.path.exists(create_some_step1_path(tmpdir))
    assert os.path.exists(create_pipeline2_path(tmpdir))
    assert os.path.exists(create_some_step2_path(tmpdir))
    assert os.path.exists(create_some_step3_path(tmpdir))


def test_resumable_pipeline_fit_transform_should_load_all_pipeline_steps(tmpdir: LocalPath):
    # Given
    root = ResumablePipeline([])
    root.sub_steps_savers = [
        (SOME_STEP_1, []),
        (PIPELINE_2, [TruncableJoblibStepSaver()])
    ]

    pipeline_2 = ResumablePipeline([])
    pipeline_2.sub_steps_savers = [
        (SOME_STEP_2, []),
        (CHECKPOINT, []),
        (SOME_STEP_3, []),
    ]

    dump(root, create_root_path(tmpdir, True))
    dump(MultiplyBy(multiply_by=2), create_some_step1_path(tmpdir, True))
    dump(pipeline_2, create_pipeline2_path(tmpdir, True))
    dump(MultiplyBy(multiply_by=4), create_some_step2_path(tmpdir, True))
    dump(MultiplyBy(multiply_by=6), create_some_step3_path(tmpdir, True))

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
