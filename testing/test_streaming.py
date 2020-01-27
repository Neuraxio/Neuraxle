import os

import numpy as np

from neuraxle.base import NonFittableMixin, BaseStep, ExecutionContext
from neuraxle.distributed.streaming import SequentialQueuedPipeline, ParallelQueuedPipeline
from neuraxle.steps.misc import FitTransformCallbackStep, TapeCallbackFunction

EXPECTED_OUTPUTS = np.array(range(100)) * 2 * 2 * 2 * 2
EXPECTED_OUTPUTS_PARALLEL = np.array((np.array(range(100)) * 2).tolist() * 4)


class MultiplyBy(NonFittableMixin, BaseStep):
    def __init__(self, multiply_by=1):
        NonFittableMixin.__init__(self)
        BaseStep.__init__(self)
        self.multiply_by = multiply_by

    def transform(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs * self.multiply_by

    def inverse_transform(self, data_inputs):
        if not isinstance(data_inputs, np.ndarray):
            data_inputs = np.array(data_inputs)

        return data_inputs / self.multiply_by


def test_queued_pipeline_with_step_incomplete_batch():
    p = SequentialQueuedPipeline([
        MultiplyBy(2),
        MultiplyBy(2),
        MultiplyBy(2),
        MultiplyBy(2)
    ], batch_size=10, n_workers_per_step=1, max_size=5)

    p, outputs = p.fit_transform(list(range(15)), list(range(15)))

    assert np.array_equal(outputs, np.array(range(15)) * 2 * 2 * 2 * 2)


def test_queued_pipeline_with_step():
    p = SequentialQueuedPipeline([
        MultiplyBy(2),
        MultiplyBy(2),
        MultiplyBy(2),
        MultiplyBy(2)
    ], batch_size=10, n_workers_per_step=1, max_size=5)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_step():
    p = SequentialQueuedPipeline([
        ('1', MultiplyBy(2)),
        ('2', MultiplyBy(2)),
        ('3', MultiplyBy(2)),
        ('4', MultiplyBy(2))
    ], batch_size=10, n_workers_per_step=1, max_size=5)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_n_workers_step():
    p = SequentialQueuedPipeline([
        (1, MultiplyBy(2)),
        (1, MultiplyBy(2)),
        (1, MultiplyBy(2)),
        (1, MultiplyBy(2))
    ], batch_size=10, max_size=5)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_max_size():
    p = SequentialQueuedPipeline([
        ('1', 1, 5, MultiplyBy(2)),
        ('2', 1, 5, MultiplyBy(2)),
        ('3', 1, 5, MultiplyBy(2)),
        ('4', 1, 5, MultiplyBy(2))
    ], batch_size=10)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_size():
    p = SequentialQueuedPipeline([
        ('1', 1, MultiplyBy(2)),
        ('2', 1, MultiplyBy(2)),
        ('3', 1, MultiplyBy(2)),
        ('4', 1, MultiplyBy(2))
    ], max_size=10, batch_size=10)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_size():
    p = SequentialQueuedPipeline([
        ('1', MultiplyBy(2)),
        ('2', MultiplyBy(2)),
        ('3', MultiplyBy(2)),
        ('4', MultiplyBy(2))
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_parallel_queued_pipeline_with_step_name_n_worker_max_size():
    p = ParallelQueuedPipeline([
        ('1', 1, 5, MultiplyBy(2)),
        ('2', 1, 5, MultiplyBy(2)),
        ('3', 1, 5, MultiplyBy(2)),
        ('4', 1, 5, MultiplyBy(2))
    ], batch_size=10)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments_max_size():
    n_workers = 4
    worker_arguments = [('multiply_by', 2) for _ in range(n_workers)]
    p = ParallelQueuedPipeline([
        ('1', n_workers, worker_arguments, 5, MultiplyBy()),
    ], batch_size=10)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    expected = np.array(list(range(0, 200, 2)))
    assert np.array_equal(outputs, expected)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments():
    n_workers = 4
    worker_arguments = [('multiply_by', 2) for _ in range(n_workers)]
    p = ParallelQueuedPipeline([
        ('1', n_workers, worker_arguments, MultiplyBy()),
    ], batch_size=10, max_size=5)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    expected = np.array(list(range(0, 200, 2)))
    assert np.array_equal(outputs, expected)


def test_parallel_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_size():
    p = ParallelQueuedPipeline([
        ('1', 1, MultiplyBy(2)),
        ('2', 1, MultiplyBy(2)),
        ('3', 1, MultiplyBy(2)),
        ('4', 1, MultiplyBy(2)),
    ], max_size=10, batch_size=10)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_size():
    p = ParallelQueuedPipeline([
        ('1', MultiplyBy(2)),
        ('2', MultiplyBy(2)),
        ('3', MultiplyBy(2)),
        ('4', MultiplyBy(2)),
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_step_name_n_worker_with_default_n_workers_and_default_max_size():
    p = ParallelQueuedPipeline([
        ('1', MultiplyBy(2)),
        ('2', MultiplyBy(2)),
        ('3', MultiplyBy(2)),
        ('4', MultiplyBy(2)),
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_queued_pipeline_saving(tmpdir):
    # Given
    p = ParallelQueuedPipeline([
        ('1', FitTransformCallbackStep()),
        ('2', FitTransformCallbackStep()),
        ('3', FitTransformCallbackStep()),
        ('4', FitTransformCallbackStep()),
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    # When
    p, outputs = p.fit_transform(list(range(100)), list(range(100)))
    p.save(ExecutionContext(tmpdir))
    p.apply('clear_callbacks')

    # Then

    assert len(p[0].wrapped.transform_callback_function.data) == 0
    assert len(p[0].wrapped.fit_callback_function.data) == 0
    assert len(p[1].wrapped.transform_callback_function.data) == 0
    assert len(p[1].wrapped.fit_callback_function.data) == 0
    assert len(p[2].wrapped.transform_callback_function.data) == 0
    assert len(p[2].wrapped.fit_callback_function.data) == 0
    assert len(p[3].wrapped.transform_callback_function.data) == 0
    assert len(p[3].wrapped.fit_callback_function.data) == 0

    p = p.load(ExecutionContext(tmpdir))

    assert len(p[0].wrapped.transform_callback_function.data) == 10
    assert len(p[0].wrapped.fit_callback_function.data) == 10
    assert len(p[1].wrapped.transform_callback_function.data) == 10
    assert len(p[1].wrapped.fit_callback_function.data) == 10
    assert len(p[2].wrapped.transform_callback_function.data) == 10
    assert len(p[2].wrapped.fit_callback_function.data) == 10
    assert len(p[3].wrapped.transform_callback_function.data) == 10
    assert len(p[3].wrapped.fit_callback_function.data) == 10

