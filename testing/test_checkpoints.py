import os
from pickle import dump

from neuraxle.checkpoints import MiniCheckpoint
from neuraxle.pipeline import ResumablePipeline
from neuraxle.steps.misc import FitTransformCallbackStep, TapeCallbackFunction


def test_resumable_pipeline_with_checkpoint_fit_transform_should_save_data_inputs(tmpdir):
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [0, 1])

    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/0_di.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/1_di.pickle'))


def test_resumable_pipeline_with_checkpoint_transform_should_save_data_inputs(tmpdir):
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.transform([0, 1])

    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/0_di.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/1_di.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_should_save_data_inputs(tmpdir):
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [0, 1])

    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/0_di.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/1_di.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_transform_should_save_expected_outputs(tmpdir):
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [0, 1])

    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/0_eo.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/1_eo.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_should_save_expected_outputs(tmpdir):
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [0, 1])

    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/0_eo.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/1_eo.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_transform_should_resume_saved_checkpoints(tmpdir):
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [0, 1])

    assert test_case.tape_transform_step1.data == []
    assert test_case.tape_transform_step2.data == [([0, 1], [0, 1])]


def test_resumable_pipeline_with_checkpoint_transform_should_resume_saved_checkpoints(tmpdir):
    given_fully_saved_checkpoints(tmpdir)
    test_case = create_checkpoint_test_case(tmpdir)

    outputs = test_case.pipeline.transform([0, 1])

    assert test_case.tape_transform_step1.data == []
    assert test_case.tape_transform_step2.data == [([0, 1], [0, 1])]


def test_resumable_pipeline_with_checkpoint_fit_should_resume_saved_checkpoints(tmpdir):
    given_fully_saved_checkpoints(tmpdir)
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [0, 1])

    assert test_case.tape_fit_step1.data == []
    assert test_case.tape_fit_step2.data == [([0, 1], [0, 1])]


def given_fully_saved_checkpoints(tmpdir):
    os.makedirs(os.path.join(tmpdir, 'ResumablePipeline/checkpoint'))
    with open(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/0_di.pickle'), 'wb') as file:
        dump(0, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/1_di.pickle'), 'wb') as file:
        dump(1, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/0_eo.pickle'), 'wb') as file:
        dump(0, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/1_eo.pickle'), 'wb') as file:
        dump(1, file)


def test_resumable_pipeline_with_checkpoint_fit_transform_should_not_resume_partially_saved_checkpoints(tmpdir):
    given_partially_saved_checkpoints(tmpdir)
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [0, 1])

    assert test_case.tape_fit_step1.data == [([0, 1], [0, 1])]
    assert test_case.tape_fit_step2.data == [([0, 1], [0, 1])]


def test_resumable_pipeline_with_checkpoint_fit_should_not_resume_partially_saved_checkpoints(tmpdir):
    given_partially_saved_checkpoints(tmpdir)
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [0, 1])

    assert test_case.tape_fit_step1.data == [([0, 1], [0, 1])]
    assert test_case.tape_fit_step2.data == [([0, 1], [0, 1])]


def test_resumable_pipeline_with_checkpoint_transform_should_not_resume_partially_saved_checkpoints(tmpdir):
    given_partially_saved_checkpoints(tmpdir)
    test_case = create_checkpoint_test_case(tmpdir)

    outputs = test_case.pipeline.transform([0, 1])

    assert test_case.tape_transform_step1.data == [[0, 1]]
    assert test_case.tape_transform_step2.data == [[0, 1]]


def given_partially_saved_checkpoints(tmpdir):
    os.makedirs(os.path.join(tmpdir, 'ResumablePipeline/checkpoint'))
    with open(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/0_di.pickle'), 'wb') as file:
        dump(0, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline/checkpoint/0_eo.pickle'), 'wb') as file:
        dump(0, file)


class CheckpointTest:
    def __init__(self, tape_transform_step1, tape_fit_step1, tape_transform_step2, tape_fit_step2, pipeline):
        self.pipeline = pipeline
        self.tape_transform_step1 = tape_transform_step1
        self.tape_fit_step1 = tape_fit_step1

        self.tape_transform_step2 = tape_transform_step2
        self.tape_fit_step2 = tape_fit_step2


def create_checkpoint_test_case(tmpdir):
    tape_transform_1 = TapeCallbackFunction()
    tape_fit_1 = TapeCallbackFunction()
    tape_transform_2 = TapeCallbackFunction()
    tape_fit_2 = TapeCallbackFunction()
    pipeline = ResumablePipeline([
        ('step1', FitTransformCallbackStep(tape_transform_1, tape_fit_1)),
        ('checkpoint', MiniCheckpoint()),
        ('step2', FitTransformCallbackStep(tape_transform_2, tape_fit_2))
    ], cache_folder=tmpdir)

    return CheckpointTest(
        tape_transform_1, tape_fit_1, tape_transform_2, tape_fit_2, pipeline
    )


def test_resumable_pipeline_with_checkpoint_should_save_steps():
    pass
