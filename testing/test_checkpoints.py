import os
from pickle import dump

from py._path.local import LocalPath

from neuraxle.checkpoints import DefaultCheckpoint
from neuraxle.pipeline import ResumablePipeline
from neuraxle.steps.misc import FitTransformCallbackStep, TapeCallbackFunction

SUMMARY_ID = '6e4419c1957e7772f3957d63bb41efcd'


def test_resumable_pipeline_with_checkpoint_fit_transform_should_save_data_inputs(tmpdir: LocalPath):
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [1, 2])

    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'di', '0.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'di', '1.pickle'))


def test_resumable_pipeline_with_checkpoint_transform_should_save_data_inputs(tmpdir):
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.transform([0, 1])

    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'di', '0.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'di', '1.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_should_save_data_inputs(tmpdir):
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [1, 2])

    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'di', '0.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'di', '1.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_transform_should_save_expected_outputs(tmpdir):
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [1, 2])

    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'eo', '0.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'eo', '1.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_should_save_expected_outputs(tmpdir):
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [1, 2])

    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'eo', '0.pickle'))
    assert os.path.exists(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'eo', '1.pickle'))


def test_resumable_pipeline_with_checkpoint_fit_transform_should_resume_saved_checkpoints(tmpdir):
    given_fully_saved_checkpoints(tmpdir)
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [1, 2])

    assert test_case.tape_transform_step1.data == []
    assert test_case.tape_fit_step1.data == []
    assert test_case.tape_fit_step2.data == [([0, 1], [1, 2])]
    assert test_case.tape_transform_step2.data == [[0, 1]]


def test_resumable_pipeline_with_checkpoint_transform_should_resume_saved_checkpoints(tmpdir):
    given_fully_saved_checkpoints(tmpdir)
    test_case = create_checkpoint_test_case(tmpdir)

    outputs = test_case.pipeline.transform([0, 1])

    assert test_case.tape_transform_step1.data == []
    assert test_case.tape_fit_step2.data == []
    assert test_case.tape_transform_step2.data == [[0, 1]]


def test_resumable_pipeline_with_checkpoint_fit_should_resume_saved_checkpoints(tmpdir):
    given_fully_saved_checkpoints(tmpdir)
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [1, 2])

    assert test_case.tape_fit_step1.data == []
    assert test_case.tape_transform_step1.data == []
    assert test_case.tape_transform_step2.data == []
    assert test_case.tape_fit_step2.data == [([0, 1], [1, 2])]


def given_fully_saved_checkpoints(tmpdir):
    os.makedirs(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'di'))
    os.makedirs(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'eo'))

    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint',
                           '{0}.txt'.format(SUMMARY_ID)), 'w+') as file:
        file.writelines([
            '0\n',
            '1'
        ])

    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'di', '0.pickle'), 'wb') as file:
        dump(0, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'di', '1.pickle'), 'wb') as file:
        dump(1, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'eo', '0.pickle'), 'wb') as file:
        dump(1, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'eo', '1.pickle'), 'wb') as file:
        dump(2, file)


def test_resumable_pipeline_with_checkpoint_fit_transform_should_not_resume_partially_saved_checkpoints(tmpdir):
    given_partially_saved_checkpoints(tmpdir)
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline, outputs = test_case.pipeline.fit_transform([0, 1], [1, 2])

    assert test_case.tape_fit_step1.data == [([0, 1], [1, 2])]
    assert test_case.tape_fit_step2.data == [([0, 1], [1, 2])]


def test_resumable_pipeline_with_checkpoint_fit_should_not_resume_partially_saved_checkpoints(tmpdir):
    given_partially_saved_checkpoints(tmpdir)
    test_case = create_checkpoint_test_case(tmpdir)

    pipeline = test_case.pipeline.fit([0, 1], [1, 2])

    assert test_case.tape_fit_step1.data == [([0, 1], [1, 2])]
    assert test_case.tape_transform_step1.data == [[0, 1]]
    assert test_case.tape_fit_step2.data == [([0, 1], [1, 2])]
    assert test_case.tape_transform_step2.data == []


def test_resumable_pipeline_with_checkpoint_transform_should_not_resume_partially_saved_checkpoints(tmpdir):
    given_partially_saved_checkpoints(tmpdir)
    test_case = create_checkpoint_test_case(tmpdir)

    outputs = test_case.pipeline.transform([0, 1])

    assert test_case.tape_fit_step1.data == []
    assert test_case.tape_transform_step1.data == [[0, 1]]
    assert test_case.tape_fit_step2.data == []
    assert test_case.tape_transform_step2.data == [[0, 1]]


def given_partially_saved_checkpoints(tmpdir):
    os.makedirs(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'di'))
    os.makedirs(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'eo'))

    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint',
                           '{0}.txt'.format(SUMMARY_ID)), 'w+') as file:
        file.writelines([
            '0\n',
            '1'
        ])
    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'di', '0.pickle'), 'wb') as file:
        dump(0, file)
    with open(os.path.join(tmpdir, 'ResumablePipeline', 'checkpoint', 'eo', '1.pickle'), 'wb') as file:
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
        ('checkpoint', DefaultCheckpoint()),
        ('step2', FitTransformCallbackStep(tape_transform_2, tape_fit_2))
    ], cache_folder=tmpdir)

    return CheckpointTest(
        tape_transform_1, tape_fit_1, tape_transform_2, tape_fit_2, pipeline
    )


def test_resumable_pipeline_with_checkpoint_should_save_steps():
    pass
