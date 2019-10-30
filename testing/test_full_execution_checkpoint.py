import os

from joblib import dump

from neuraxle.pipeline import ResumablePipeline, FullExecutionCheckpoint
from neuraxle.steps.misc import TapeCallbackFunction, FitTransformCallbackStep
from testing.test_checkpoints import CheckpointTest

EXPECTED_OUTPUT_CHECKPOINTER_WRAPPER = 'ExpectedOutputCheckpointerWrapper'

DATA_INPUT_CHECKPOINTER_WRAPPER = 'DataInputCheckpointerWrapper'

FIT_TRANSFORM_ONLY_WRAPPER_DI_NAME = 'FitTransformOnlyWrapperDi'
TRANSFORM_ONLY_WRAPPER_DI_NAME = 'TransformOnlyWrapperDi'
FIT_ONLY_WRAPPER_DI_NAME = 'FitOnlyWrapperDi'

FIT_TRANSFORM_ONLY_WRAPPER_EO_NAME = 'FitTransformOnlyWrapperEo'
TRANSFORM_ONLY_WRAPPER_EO_NAME = 'TransformOnlyWrapperEo'
FIT_ONLY_WRAPPER_EO_NAME = 'FitOnlyWrapperEo'

def test_resumable_pipeline_with_checkpoint_fit_transform_should_save_data_inputs(tmpdir):
    test_case = create_full_execution_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [1, 2])

    assert os.path.exists(
        os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', FIT_TRANSFORM_ONLY_WRAPPER_DI_NAME, DATA_INPUT_CHECKPOINTER_WRAPPER, '0.pickle'))
    assert os.path.exists(
        os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', FIT_TRANSFORM_ONLY_WRAPPER_DI_NAME, DATA_INPUT_CHECKPOINTER_WRAPPER, '1.pickle'))


def test_resumable_pipeline_with_checkpoint_transform_should_save_data_inputs(tmpdir):
    test_case = create_full_execution_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.transform([0, 1])

    assert os.path.exists(
        os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', TRANSFORM_ONLY_WRAPPER_DI_NAME,
                     DATA_INPUT_CHECKPOINTER_WRAPPER, '0.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', TRANSFORM_ONLY_WRAPPER_DI_NAME,
                                       DATA_INPUT_CHECKPOINTER_WRAPPER, '1.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_should_save_data_inputs(tmpdir):
    test_case = create_full_execution_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [1, 2])

    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', FIT_ONLY_WRAPPER_DI_NAME,
                                       DATA_INPUT_CHECKPOINTER_WRAPPER, '0.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', FIT_ONLY_WRAPPER_DI_NAME,
                                       DATA_INPUT_CHECKPOINTER_WRAPPER, '1.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_transform_should_save_expected_outputs(tmpdir):
    test_case = create_full_execution_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [1, 2])

    assert os.path.exists(
        os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', FIT_TRANSFORM_ONLY_WRAPPER_EO_NAME,
                     EXPECTED_OUTPUT_CHECKPOINTER_WRAPPER, '0.pickle'))
    assert os.path.exists(
        os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', FIT_TRANSFORM_ONLY_WRAPPER_EO_NAME,
                     EXPECTED_OUTPUT_CHECKPOINTER_WRAPPER, '1.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_should_save_expected_outputs(tmpdir):
    test_case = create_full_execution_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [1, 2])

    assert os.path.exists(
        os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', FIT_ONLY_WRAPPER_EO_NAME,
                     EXPECTED_OUTPUT_CHECKPOINTER_WRAPPER, '0.pickle'))
    assert os.path.exists(
        os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', FIT_ONLY_WRAPPER_EO_NAME,
                     EXPECTED_OUTPUT_CHECKPOINTER_WRAPPER, '1.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_transform_should_resume_saved_checkpoints(tmpdir):
    given_fully_saved_checkpoints(tmpdir, FIT_TRANSFORM_ONLY_WRAPPER_DI_NAME, FIT_TRANSFORM_ONLY_WRAPPER_EO_NAME)
    test_case = create_full_execution_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [1, 2])

    assert test_case.tape_transform_step1.data == []
    assert test_case.tape_fit_step1.data == []
    assert test_case.tape_fit_step2.data == [([0, 1], [1, 2])]
    assert test_case.tape_transform_step2.data == [[0, 1]]


def test_resumable_pipeline_with_checkpoint_transform_should_resume_saved_checkpoints(tmpdir):
    given_fully_saved_checkpoints(tmpdir, TRANSFORM_ONLY_WRAPPER_DI_NAME, TRANSFORM_ONLY_WRAPPER_EO_NAME)
    test_case = create_full_execution_checkpoint_test_case(tmpdir)

    outputs = test_case.pipeline.transform([0, 1])

    assert test_case.tape_transform_step1.data == []
    assert test_case.tape_fit_step2.data == []
    assert test_case.tape_transform_step2.data == [[0, 1]]


def test_resumable_pipeline_with_checkpoint_fit_should_resume_saved_checkpoints(tmpdir):
    given_fully_saved_checkpoints(tmpdir, FIT_ONLY_WRAPPER_DI_NAME, FIT_ONLY_WRAPPER_EO_NAME)
    test_case = create_full_execution_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [1, 2])

    assert test_case.tape_fit_step1.data == []
    assert test_case.tape_transform_step1.data == []
    assert test_case.tape_transform_step2.data == []
    assert test_case.tape_fit_step2.data == [([0, 1], [1, 2])]


def test_resumable_pipeline_with_checkpoint_fit_transform_should_not_resume_partially_saved_checkpoints(tmpdir):
    given_partially_saved_checkpoints(tmpdir, FIT_TRANSFORM_ONLY_WRAPPER_DI_NAME, FIT_TRANSFORM_ONLY_WRAPPER_EO_NAME)
    test_case = create_full_execution_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [1, 2])

    assert test_case.tape_fit_step1.data == [([0, 1], [1, 2])]
    assert test_case.tape_fit_step2.data == [([0, 1], [1, 2])]


def test_resumable_pipeline_with_checkpoint_fit_should_not_resume_partially_saved_checkpoints(tmpdir):
    given_partially_saved_checkpoints(tmpdir, FIT_ONLY_WRAPPER_DI_NAME, FIT_ONLY_WRAPPER_EO_NAME)
    test_case = create_full_execution_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [1, 2])

    assert test_case.tape_fit_step1.data == [([0, 1], [1, 2])]
    assert test_case.tape_transform_step1.data == [[0, 1]]
    assert test_case.tape_fit_step2.data == [([0, 1], [1, 2])]
    assert test_case.tape_transform_step2.data == []


def test_resumable_pipeline_with_checkpoint_transform_should_not_resume_partially_saved_checkpoints(tmpdir):
    given_partially_saved_checkpoints(tmpdir, TRANSFORM_ONLY_WRAPPER_DI_NAME, TRANSFORM_ONLY_WRAPPER_EO_NAME)
    test_case = create_full_execution_checkpoint_test_case(tmpdir)

    outputs = test_case.pipeline.transform([0, 1])

    assert test_case.tape_fit_step1.data == []
    assert test_case.tape_transform_step1.data == [[0, 1]]
    assert test_case.tape_fit_step2.data == []
    assert test_case.tape_transform_step2.data == [[0, 1]]


def given_partially_saved_checkpoints(tmpdir, execution_mode_di_name, execution_mode_eo_name):
    os.makedirs(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', execution_mode_di_name, DATA_INPUT_CHECKPOINTER_WRAPPER))
    os.makedirs(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', execution_mode_eo_name, EXPECTED_OUTPUT_CHECKPOINTER_WRAPPER))

    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', execution_mode_di_name,
                           DATA_INPUT_CHECKPOINTER_WRAPPER, '0.pickle'),
              'wb') as file:
        dump(0, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', execution_mode_eo_name,
                           EXPECTED_OUTPUT_CHECKPOINTER_WRAPPER, '1.pickle'),
              'wb') as file:
        dump(0, file)


def given_fully_saved_checkpoints(tmpdir, execution_mode_name_di, execution_mode_name_eo):
    os.makedirs(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', execution_mode_name_di, DATA_INPUT_CHECKPOINTER_WRAPPER))
    os.makedirs(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', execution_mode_name_eo, EXPECTED_OUTPUT_CHECKPOINTER_WRAPPER))

    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', execution_mode_name_di,
                           DATA_INPUT_CHECKPOINTER_WRAPPER, '0.pickle'),
              'wb') as file:
        dump(0, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', execution_mode_name_di,
                           DATA_INPUT_CHECKPOINTER_WRAPPER, '1.pickle'),
              'wb') as file:
        dump(1, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', execution_mode_name_eo,
                           EXPECTED_OUTPUT_CHECKPOINTER_WRAPPER, '0.pickle'),
              'wb') as file:
        dump(1, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', execution_mode_name_eo,
                           EXPECTED_OUTPUT_CHECKPOINTER_WRAPPER, '1.pickle'),
              'wb') as file:
        dump(2, file)


def create_full_execution_checkpoint_test_case(tmpdir):
    tape_transform_1 = TapeCallbackFunction()
    tape_fit_1 = TapeCallbackFunction()
    tape_transform_2 = TapeCallbackFunction()
    tape_fit_2 = TapeCallbackFunction()
    pipeline = ResumablePipeline([
        ('step1', FitTransformCallbackStep(tape_transform_1, tape_fit_1)),
        ('checkpoint', FullExecutionCheckpoint()),
        ('step2', FitTransformCallbackStep(tape_transform_2, tape_fit_2))
    ], cache_folder=tmpdir)

    return CheckpointTest(
        tape_transform_1, tape_fit_1, tape_transform_2, tape_fit_2, pipeline
    )
