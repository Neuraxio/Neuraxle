import numpy as np

from neuraxle.distributed.streaming import QueuedPipeline
from neuraxle.steps.misc import TapeCallbackFunction, FitTransformCallbackStep

EXPECTED_OUTPUTS = np.array(range(100)) * 2 * 2 * 2 * 2


class MultiplyBy2FitTransformCallbackStep(FitTransformCallbackStep):
    def fit_transform(self, data_inputs, expected_outputs=None):
        FitTransformCallbackStep.fit_transform(self, data_inputs, expected_outputs)

        return self, list(np.array(data_inputs) * 2)


def test_queued_pipeline_with_step_name_n_worker_max_size():
    # Given
    tape1 = TapeCallbackFunction()
    tape1_fit = TapeCallbackFunction()
    tape2 = TapeCallbackFunction()
    tape2_fit = TapeCallbackFunction()
    tape3 = TapeCallbackFunction()
    tape3_fit = TapeCallbackFunction()
    tape4 = TapeCallbackFunction()
    tape4_fit = TapeCallbackFunction()

    p = QueuedPipeline([
        ('1', 1, 5, MultiplyBy2FitTransformCallbackStep(tape1, tape1_fit, ["1"])),
        ('2', 1, 5, MultiplyBy2FitTransformCallbackStep(tape2, tape2_fit, ["2"])),
        ('3', 1, 5, MultiplyBy2FitTransformCallbackStep(tape3, tape3_fit, ["3"])),
        ('4', 1, 5, MultiplyBy2FitTransformCallbackStep(tape4, tape4_fit, ["4"])),
    ], batch_size=10)

    # When
    p, outputs = p.fit_transform(range(100), range(100))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_size():
    # Given
    tape1 = TapeCallbackFunction()
    tape1_fit = TapeCallbackFunction()
    tape2 = TapeCallbackFunction()
    tape2_fit = TapeCallbackFunction()
    tape3 = TapeCallbackFunction()
    tape3_fit = TapeCallbackFunction()
    tape4 = TapeCallbackFunction()
    tape4_fit = TapeCallbackFunction()

    p = QueuedPipeline([
        ('1', 1, MultiplyBy2FitTransformCallbackStep(tape1, tape1_fit, ["1"])),
        ('2', 1, MultiplyBy2FitTransformCallbackStep(tape2, tape2_fit, ["2"])),
        ('3', 1, MultiplyBy2FitTransformCallbackStep(tape3, tape3_fit, ["3"])),
        ('4', 1, MultiplyBy2FitTransformCallbackStep(tape4, tape4_fit, ["4"])),
    ], max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(range(100), range(100))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_size():
    # Given
    tape1 = TapeCallbackFunction()
    tape1_fit = TapeCallbackFunction()
    tape2 = TapeCallbackFunction()
    tape2_fit = TapeCallbackFunction()
    tape3 = TapeCallbackFunction()
    tape3_fit = TapeCallbackFunction()
    tape4 = TapeCallbackFunction()
    tape4_fit = TapeCallbackFunction()

    p = QueuedPipeline([
        ('1', MultiplyBy2FitTransformCallbackStep(tape1, tape1_fit, ["1"])),
        ('2', MultiplyBy2FitTransformCallbackStep(tape2, tape2_fit, ["2"])),
        ('3', MultiplyBy2FitTransformCallbackStep(tape3, tape3_fit, ["3"])),
        ('4', MultiplyBy2FitTransformCallbackStep(tape4, tape4_fit, ["4"])),
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(range(100), range(100))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)
