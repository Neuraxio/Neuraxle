import os

import numpy as np
from joblib import dump
from py._path.local import LocalPath

from neuraxle.base import BaseStep, TruncableJoblibStepSaver, NonFittableMixin
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

def create_some_checkpoint_path(tmpdir, create_dir=False):
    p = os.path.join(tmpdir, ROOT, PIPELINE_2, CHECKPOINT, '{0}.joblib'.format(CHECKPOINT))
    if create_dir:
        os.makedirs(os.path.join(tmpdir, ROOT, PIPELINE_2, CHECKPOINT))
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
    if create_dir and not os.path.exists(os.path.join(tmpdir, ROOT)):
        os.makedirs(os.path.join(tmpdir, ROOT))
    return p


class MultiplyBy(NonFittableMixin, BaseStep):
    def __init__(self, multiply_by):
        BaseStep.__init__(self)
        self.multiply_by = multiply_by

    def transform(self, data_inputs):
        return data_inputs * self.multiply_by


def test_resumable_pipeline_fit_transform_should_save_all_fitted_pipeline_steps(tmpdir: LocalPath):
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

    not_saved_paths = [create_some_step3_path(tmpdir)]
    saved_paths = [create_root_path(tmpdir), create_pipeline2_path(tmpdir), create_some_step1_path(tmpdir),
                   create_some_step2_path(tmpdir), create_some_checkpoint_path(tmpdir)]
    assert np.array_equal(outputs, EXPECTED_OUTPUTS)
    for p in saved_paths:
        assert os.path.exists(p)
    for p in not_saved_paths:
        assert not os.path.exists(p)


def test_resumable_pipeline_transform_should_not_save_steps(tmpdir: LocalPath):
    p = ResumablePipeline([
        (SOME_STEP_1, MultiplyBy(multiply_by=2)),
        (PIPELINE_2, ResumablePipeline([
            (SOME_STEP_2, MultiplyBy(multiply_by=4)),
            (CHECKPOINT, PickleCheckpointStep(cache_folder=tmpdir)),
            (SOME_STEP_3, MultiplyBy(multiply_by=6)),
        ]))
    ], cache_folder=tmpdir)
    p.name = ROOT

    outputs = p.transform(
        np.array(range(10))
    )

    saved_paths = [create_root_path(tmpdir), create_pipeline2_path(tmpdir), create_some_step1_path(tmpdir),
             create_some_step2_path(tmpdir), create_some_step3_path(tmpdir), create_some_checkpoint_path(tmpdir)]
    assert np.array_equal(outputs, EXPECTED_OUTPUTS)
    for p in saved_paths:
        assert not os.path.exists(p)


def test_resumable_pipeline_fit_should_save_all_fitted_pipeline_steps(tmpdir: LocalPath):
    p = ResumablePipeline([
        (SOME_STEP_1, MultiplyBy(multiply_by=2)),
        (PIPELINE_2, ResumablePipeline([
            (SOME_STEP_2, MultiplyBy(multiply_by=4)),
            (CHECKPOINT, PickleCheckpointStep(cache_folder=tmpdir)),
            (SOME_STEP_3, MultiplyBy(multiply_by=6)),
        ]))
    ], cache_folder=tmpdir)
    p.name = ROOT

    p = p.fit(
        np.array(range(10)),
        np.array(range(10))
    )

    not_saved_paths = [create_some_step3_path(tmpdir)]
    saved_paths = [create_root_path(tmpdir), create_pipeline2_path(tmpdir), create_some_step1_path(tmpdir),
                   create_some_step2_path(tmpdir), create_some_checkpoint_path(tmpdir)]
    for p in saved_paths:
        assert os.path.exists(p)
    for p in not_saved_paths:
        assert not os.path.exists(p)


def test_resumable_pipeline_fit_transform_should_load_all_pipeline_steps(tmpdir: LocalPath):
    # Given
    p = given_saved_pipeline(tmpdir)

    # When
    p, outputs = p.fit_transform(
        np.array(range(10)),
        np.array(range(10))
    )

    # Then
    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_resumable_pipeline_transform_should_load_all_pipeline_steps(tmpdir: LocalPath):
    # Given
    p = given_saved_pipeline(tmpdir)

    # When
    outputs = p.transform(
        np.array(range(10))
    )

    # Then
    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_resumable_pipeline_fit_should_load_all_pipeline_steps(tmpdir: LocalPath):
    # Given
    p = given_saved_pipeline(tmpdir)

    # When
    p = p.fit(
        np.array(range(10)),
        np.array(range(10))
    )

    # Then
    assert p[SOME_STEP_1].multiply_by == 2
    assert p[PIPELINE_2][SOME_STEP_2].multiply_by == 4
    assert p[PIPELINE_2][SOME_STEP_3].multiply_by == 6


def given_saved_pipeline(tmpdir):
    step_savers = [(SOME_STEP_1, []), (PIPELINE_2, [TruncableJoblibStepSaver()])]
    path = create_root_path(tmpdir, True)
    root = ResumablePipeline([], cache_folder=tmpdir)
    root.sub_steps_savers = step_savers
    root.name = ROOT
    dump(root, path)

    pipeline_2 = ResumablePipeline([], cache_folder=tmpdir)
    pipeline_2.name = 'pipeline2'
    pipeline_2.sub_steps_savers = [
        (SOME_STEP_2, []),
        (CHECKPOINT, []),
        (SOME_STEP_3, []),
    ]
    dump(pipeline_2, create_pipeline2_path(tmpdir, True))

    given_saved_some_step(multiply_by=2, name=SOME_STEP_1, path=create_some_step1_path(tmpdir, True))
    given_saved_some_step(multiply_by=4, name=SOME_STEP_2, path=create_some_step2_path(tmpdir, True))
    given_saved_some_step(multiply_by=6, name=SOME_STEP_3, path=create_some_step3_path(tmpdir, True))

    pickle_checkpoint_step = PickleCheckpointStep(cache_folder=tmpdir)
    pickle_checkpoint_step.name = CHECKPOINT
    dump(pickle_checkpoint_step, create_some_checkpoint_path(tmpdir, True))

    p = ResumablePipeline([
        (SOME_STEP_1, MultiplyBy(multiply_by=1)),
        (PIPELINE_2, ResumablePipeline([
            (SOME_STEP_2, MultiplyBy(multiply_by=1)),
            (CHECKPOINT, PickleCheckpointStep(cache_folder=tmpdir)),
            (SOME_STEP_3, MultiplyBy(multiply_by=1)),
        ]))
    ], cache_folder=tmpdir)
    p.name = ROOT

    return p


def given_saved_some_step(multiply_by, name, path):
    some_step1 = MultiplyBy(multiply_by=multiply_by)
    some_step1.name = name
    dump(some_step1, path)
