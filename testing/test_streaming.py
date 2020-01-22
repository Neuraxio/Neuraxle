import numpy as np

from neuraxle.distributed.streaming import SequentialQueuedPipeline, ParallelQueuedPipeline
from neuraxle.steps.misc import TapeCallbackFunction, FitTransformCallbackStep
from neuraxle.steps.numpy import MultiplyByN

EXPECTED_OUTPUTS = np.array(range(100)) * 2 * 2 * 2 * 2
EXPECTED_OUTPUTS_PARALLEL = np.array((np.array(range(100)) * 2).tolist() * 4)


class MultiplyBy2CallbackStep(FitTransformCallbackStep):
    def __init__(self, tape_transform, tape_fit, more_arguments):
        FitTransformCallbackStep.__init__(
            self,
            transform_callback_function=tape_transform,
            fit_callback_function=tape_fit,
            more_arguments=more_arguments,
            transform_function=multiply_by_2
        )


def multiply_by_2(di):
    return np.array(di) * 2


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

    p = SequentialQueuedPipeline([
        ('1', 1, 5, MultiplyBy2CallbackStep(tape1, tape1_fit, ["1"])),
        ('2', 1, 5, MultiplyBy2CallbackStep(tape2, tape2_fit, ["2"])),
        ('3', 1, 5, MultiplyBy2CallbackStep(tape3, tape3_fit, ["3"])),
        ('4', 1, 5, MultiplyBy2CallbackStep(tape4, tape4_fit, ["4"]))
    ], batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

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

    p = SequentialQueuedPipeline([
        ('1', 1, MultiplyBy2CallbackStep(tape1, tape1_fit, ["1"])),
        ('2', 1, MultiplyBy2CallbackStep(tape2, tape2_fit, ["2"])),
        ('3', 1, MultiplyBy2CallbackStep(tape3, tape3_fit, ["3"])),
        ('4', 1, MultiplyBy2CallbackStep(tape4, tape4_fit, ["4"]))
    ], max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

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

    p = SequentialQueuedPipeline([
        ('1', MultiplyBy2CallbackStep(tape1, tape1_fit, ["1"])),
        ('2', MultiplyBy2CallbackStep(tape2, tape2_fit, ["2"])),
        ('3', MultiplyBy2CallbackStep(tape3, tape3_fit, ["3"])),
        ('4', MultiplyBy2CallbackStep(tape4, tape4_fit, ["4"]))
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_parallel_queued_pipeline_with_step_name_n_worker_max_size():
    # Given
    tape1 = TapeCallbackFunction()
    tape1_fit = TapeCallbackFunction()
    tape2 = TapeCallbackFunction()
    tape2_fit = TapeCallbackFunction()
    tape3 = TapeCallbackFunction()
    tape3_fit = TapeCallbackFunction()
    tape4 = TapeCallbackFunction()
    tape4_fit = TapeCallbackFunction()

    p = ParallelQueuedPipeline([
        ('1', 1, 5, MultiplyBy2CallbackStep(tape1, tape1_fit, ["1"])),
        ('2', 1, 5, MultiplyBy2CallbackStep(tape2, tape2_fit, ["2"])),
        ('3', 1, 5, MultiplyBy2CallbackStep(tape3, tape3_fit, ["3"])),
        ('4', 1, 5, MultiplyBy2CallbackStep(tape4, tape4_fit, ["4"]))
    ], batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments_max_size():
    n_workers = 4
    worker_arguments = [('multiply_by', 2) for _ in range(n_workers)]
    p = ParallelQueuedPipeline([
        ('1', n_workers, worker_arguments, 5, MultiplyByN()),
    ], batch_size=10)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments():
    n_workers = 4
    worker_arguments = [('multiply_by', 2) for _ in range(n_workers)]
    p = ParallelQueuedPipeline([
        ('1', n_workers, worker_arguments, MultiplyByN()),
    ], batch_size=10, max_size=5)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_size():
    # Given
    tape1 = TapeCallbackFunction()
    tape1_fit = TapeCallbackFunction()
    tape2 = TapeCallbackFunction()
    tape2_fit = TapeCallbackFunction()
    tape3 = TapeCallbackFunction()
    tape3_fit = TapeCallbackFunction()
    tape4 = TapeCallbackFunction()
    tape4_fit = TapeCallbackFunction()

    p = ParallelQueuedPipeline([
        ('1', 1, MultiplyBy2CallbackStep(tape1, tape1_fit, ["1"])),
        ('2', 1, MultiplyBy2CallbackStep(tape2, tape2_fit, ["2"])),
        ('3', 1, MultiplyBy2CallbackStep(tape3, tape3_fit, ["3"])),
        ('4', 1, MultiplyBy2CallbackStep(tape4, tape4_fit, ["4"])),
    ], max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_size():
    # Given
    tape1 = TapeCallbackFunction()
    tape1_fit = TapeCallbackFunction()
    tape2 = TapeCallbackFunction()
    tape2_fit = TapeCallbackFunction()
    tape3 = TapeCallbackFunction()
    tape3_fit = TapeCallbackFunction()
    tape4 = TapeCallbackFunction()
    tape4_fit = TapeCallbackFunction()

    p = ParallelQueuedPipeline([
        ('1', MultiplyBy2CallbackStep(tape1, tape1_fit, ["1"])),
        ('2', MultiplyBy2CallbackStep(tape2, tape2_fit, ["2"])),
        ('3', MultiplyBy2CallbackStep(tape3, tape3_fit, ["3"])),
        ('4', MultiplyBy2CallbackStep(tape4, tape4_fit, ["4"])),
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_step_name_n_worker_with_default_n_workers_and_default_max_size():
    # Given
    tape1 = TapeCallbackFunction()
    tape1_fit = TapeCallbackFunction()
    tape2 = TapeCallbackFunction()
    tape2_fit = TapeCallbackFunction()
    tape3 = TapeCallbackFunction()
    tape3_fit = TapeCallbackFunction()
    tape4 = TapeCallbackFunction()
    tape4_fit = TapeCallbackFunction()

    p = ParallelQueuedPipeline([
        ('1', MultiplyBy2CallbackStep(tape1, tape1_fit, ["1"])),
        ('2', MultiplyBy2CallbackStep(tape2, tape2_fit, ["2"])),
        ('3', MultiplyBy2CallbackStep(tape3, tape3_fit, ["3"])),
        ('4', MultiplyBy2CallbackStep(tape4, tape4_fit, ["4"])),
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)
