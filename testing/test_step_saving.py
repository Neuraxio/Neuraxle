import os

import numpy as np
from joblib import dump
from py._path.local import LocalPath

from neuraxle.base import TruncableJoblibStepSaver
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN

OUTPUT = "OUTPUT"
ROOT = 'Pipeline'
PIPELINE_2 = 'Pipeline2'
SOME_STEPS = ['some_step0', 'some_step1', 'some_step2']

EXPECTED_OUTPUTS = [0, 48, 96, 144, 192, 240, 288, 336, 384, 432]


def create_some_step_path(tmpdir, step_no=0, create_dir=False):
    path1 = os.path.join(tmpdir, ROOT, PIPELINE_2, SOME_STEPS[step_no])
    if create_dir:
        os.makedirs(path1)
    path2 = os.path.join(path1, '{0}.joblib'.format(SOME_STEPS[step_no]))
    return path2


def create_pipeline2_path(tmpdir, create_dir=False):
    path1 = os.path.join(tmpdir, ROOT, PIPELINE_2)
    if create_dir:
        os.makedirs(path1)
    path2 = os.path.join(path1, '{0}.joblib'.format(PIPELINE_2))
    return path2


def create_root_path(tmpdir, create_dir=False):
    path1 = os.path.join(tmpdir, ROOT)
    if create_dir and not os.path.exists(os.path.join(tmpdir, ROOT)):
        os.makedirs(path1)
    path2 = os.path.join(path1, '{0}.joblib'.format(ROOT))
    return path2


def test_nested_pipeline_fit_transform_should_save_all_fitted_pipeline_steps(tmpdir: LocalPath):
    p = Pipeline([
        (SOME_STEPS[1], MultiplyByN(multiply_by=2)),
        (PIPELINE_2, Pipeline([
            (SOME_STEPS[2], MultiplyByN(multiply_by=4)),
            (SOME_STEPS[3], MultiplyByN(multiply_by=6))
        ]))
    ], cache_folder=tmpdir)
    p.name = ROOT

    p, outputs = p.fit_transform(
        np.array(range(10)),
        np.array(range(10))
    )

    not_saved_paths = [create_some_step_path(tmpdir, step_no=3)]
    saved_paths = [create_root_path(tmpdir), create_pipeline2_path(tmpdir), create_some_step_path(tmpdir, step_no=1),
                   create_some_step_path(tmpdir, step_no=2)]
    assert np.array_equal(outputs, EXPECTED_OUTPUTS)
    for p in saved_paths:
        assert os.path.exists(p)
    for p in not_saved_paths:
        assert not os.path.exists(p)


def test_resumable_pipeline_transform_should_not_save_steps(tmpdir: LocalPath):
    p = Pipeline([
        (SOME_STEPS[1], MultiplyByN(multiply_by=2)),
        (PIPELINE_2, Pipeline([
            (SOME_STEPS[2], MultiplyByN(multiply_by=4)),
            (SOME_STEPS[3], MultiplyByN(multiply_by=6)),
        ]))
    ], cache_folder=tmpdir)
    p.name = ROOT

    outputs = p.transform(
        np.array(range(10))
    )

    saved_paths = [create_root_path(tmpdir), create_pipeline2_path(tmpdir), create_some_step_path(tmpdir, step_no=1),
                   create_some_step_path(tmpdir, step_no=2), create_some_step_path(tmpdir, step_no=3)]
    assert np.array_equal(outputs, EXPECTED_OUTPUTS)
    for p in saved_paths:
        assert not os.path.exists(p)


def test_resumable_pipeline_fit_should_save_all_fitted_pipeline_steps(tmpdir: LocalPath):
    p = Pipeline([
        (SOME_STEPS[1], MultiplyByN(multiply_by=2)),
        (PIPELINE_2, Pipeline([
            (SOME_STEPS[2], MultiplyByN(multiply_by=4)),
            (SOME_STEPS[3], MultiplyByN(multiply_by=6))
        ]))
    ], cache_folder=tmpdir)
    p.name = ROOT

    p = p.fit(
        np.array(range(10)),
        np.array(range(10))
    )

    not_saved_paths = [create_some_step_path(tmpdir, step_no=3)]
    saved_paths = [create_root_path(tmpdir), create_pipeline2_path(tmpdir), create_some_step_path(tmpdir, step_no=1),
                   create_some_step_path(tmpdir, step_no=2)]
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
    assert p[SOME_STEPS[1]].hyperparams['multiply_by'] == 2
    assert p[PIPELINE_2][SOME_STEPS[2]].hyperparams['multiply_by'] == 4
    assert p[PIPELINE_2][SOME_STEPS[3]].hyperparams['multiply_by'] == 6


def given_saved_pipeline(tmpdir: LocalPath):
    step_savers = [(SOME_STEPS[1], []), (PIPELINE_2, [TruncableJoblibStepSaver()])]
    path = create_root_path(tmpdir, True)
    root = Pipeline([], cache_folder=tmpdir)
    root.sub_steps_savers = step_savers
    root.name = ROOT
    dump(root, path)

    pipeline_2 = Pipeline([], cache_folder=tmpdir)
    pipeline_2.name = 'pipeline2'
    pipeline_2.sub_steps_savers = [
        (SOME_STEPS[2], []),
        (CHECKPOINT, []),
        (SOME_STEPS[3], []),
    ]
    dump(pipeline_2, create_pipeline2_path(tmpdir, True))

    given_saved_some_step(
        multiply_by=2, name=SOME_STEPS[1], path=create_some_step_path(tmpdir, step_no=1, create_dir=True))
    given_saved_some_step(
        multiply_by=4, name=SOME_STEPS[2], path=create_some_step_path(tmpdir, step_no=2, create_dir=True))
    given_saved_some_step(
        multiply_by=6, name=SOME_STEPS[3], path=create_some_step_path(tmpdir, step_no=3, create_dir=True))

    p = Pipeline([
        (SOME_STEPS[1], MultiplyByN(multiply_by=1)),
        (PIPELINE_2, Pipeline([
            (SOME_STEPS[2], MultiplyByN(multiply_by=1)),
            (SOME_STEPS[3], MultiplyByN(multiply_by=1))
        ]))
    ], cache_folder=tmpdir)
    p.name = ROOT

    return p


def given_saved_some_step(multiply_by, name, path):
    some_step1 = MultiplyByN(multiply_by=multiply_by)
    some_step1.name = name
    dump(some_step1, path)
