import os

import numpy as np
from joblib import dump
from py._path.local import LocalPath
from pprint import pprint

from neuraxle.hyperparams.space import RecursiveDict
from neuraxle.base import CX, StepWithContext, TruncableJoblibStepSaver
from neuraxle.pipeline import Pipeline
from neuraxle.steps.numpy import MultiplyByN

OUTPUT = "OUTPUT"
ROOT = 'Pipeline'
PIPELINE_2 = 'Pipeline2'
SOME_STEPS = ['some_step0', 'some_step1', 'some_step2']

EXPECTED_OUTPUTS = [0, 48, 96, 144, 192, 240, 288, 336, 384, 432]


def create_some_step_path(tmpdir, step_no=0, create_dir=False):
    if step_no == 0:
        path1 = os.path.join(tmpdir, ROOT, SOME_STEPS[step_no])
    else:
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


def test_nested_pipeline_fit_transform_should_save_some_fitted_pipeline_steps(tmpdir: LocalPath):
    p: StepWithContext = create_pipeline(tmpdir)

    p, outputs = p.fit_transform(np.array(range(10)), np.array(range(10)))
    p.save()

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)
    saved_paths = [
        create_root_path(tmpdir), create_pipeline2_path(tmpdir),
        create_some_step_path(tmpdir, step_no=0), create_some_step_path(tmpdir, step_no=1),
        create_some_step_path(tmpdir, step_no=2)
    ]
    for path in saved_paths:
        assert os.path.exists(path), path


def test_pipeline_transform_should_not_save_steps(tmpdir: LocalPath):
    p: StepWithContext = create_pipeline(tmpdir)

    outputs = p.transform(np.array(range(10)))
    p.wrapped.save(CX(tmpdir), full_dump=False)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)
    not_saved_paths = [
        create_root_path(tmpdir), create_pipeline2_path(tmpdir), create_some_step_path(tmpdir, step_no=0),
        create_some_step_path(tmpdir, step_no=1), create_some_step_path(tmpdir, step_no=2)]
    for path in not_saved_paths:
        assert not os.path.exists(path), path


def test_pipeline_fit_should_save_all_fitted_pipeline_steps(tmpdir: LocalPath):
    p: StepWithContext = create_pipeline(tmpdir)

    p = p.fit(np.array(range(10)), np.array(range(10)))
    p.save()

    saved_paths = [
        create_root_path(tmpdir), create_pipeline2_path(tmpdir),
        create_some_step_path(tmpdir, step_no=0), create_some_step_path(tmpdir, step_no=1),
        create_some_step_path(tmpdir, step_no=2)
    ]
    for path in saved_paths:
        assert os.path.exists(path), path


def test_pipeline_fit_transform_should_load_all_pipeline_steps(tmpdir: LocalPath):
    p = given_saved_pipeline(tmpdir)

    p, outputs = p.fit_transform(np.array(range(10)), np.array(range(10)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_pipeline_transform_should_load_all_pipeline_steps(tmpdir: LocalPath):
    p = given_saved_pipeline(tmpdir)

    outputs = p.transform(np.array(range(10)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_pipeline_fit_should_load_all_pipeline_steps(tmpdir: LocalPath):
    p = given_saved_pipeline(tmpdir)

    p = p.fit(np.array(range(10)), np.array(range(10)))

    assert p.wrapped[SOME_STEPS[0]].hyperparams['multiply_by'] == 2
    assert p.wrapped[PIPELINE_2][SOME_STEPS[1]].hyperparams['multiply_by'] == 4
    assert p.wrapped[PIPELINE_2][SOME_STEPS[2]].hyperparams['multiply_by'] == 6


def given_saved_pipeline(tmpdir: LocalPath) -> Pipeline:
    path = create_root_path(tmpdir, True)
    p = Pipeline([]).set_name(ROOT).with_context(CX(tmpdir)).with_context(CX(tmpdir))
    dump(p, path)

    pipeline_2 = Pipeline([]).set_name(PIPELINE_2).with_context(CX(tmpdir))
    pipeline_2.sub_steps_savers = [
        (SOME_STEPS[0], []),
        (SOME_STEPS[1], []),
    ]
    dump(pipeline_2, create_pipeline2_path(tmpdir, True))

    given_saved_some_step(multiply_by=2, step_no=0, path=create_some_step_path(tmpdir, step_no=0, create_dir=True))
    given_saved_some_step(multiply_by=4, step_no=1, path=create_some_step_path(tmpdir, step_no=1, create_dir=True))
    given_saved_some_step(multiply_by=6, step_no=2, path=create_some_step_path(tmpdir, step_no=2, create_dir=True))

    p = create_pipeline(tmpdir)

    return p


def create_pipeline(tmpdir) -> StepWithContext:
    return Pipeline([
        (SOME_STEPS[0], MultiplyByN(multiply_by=2)),
        (PIPELINE_2, Pipeline([
            (SOME_STEPS[1], MultiplyByN(multiply_by=4)),
            (SOME_STEPS[2], MultiplyByN(multiply_by=6))
        ]))
    ]).set_name(ROOT).with_context(CX(tmpdir))


def given_saved_some_step(multiply_by, step_no, path):
    some_step1 = MultiplyByN(multiply_by=multiply_by)
    some_step1.name = SOME_STEPS[step_no]
    dump(some_step1, path)
