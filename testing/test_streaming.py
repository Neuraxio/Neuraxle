import time

import numpy as np

from neuraxle.base import BaseStep, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.distributed.streaming import SequentialQueuedPipeline, ParallelQueuedFeatureUnion, QueueJoiner
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import ForEachDataInput
from neuraxle.steps.misc import FitTransformCallbackStep, Sleep

EXPECTED_OUTPUTS = np.array(range(100)) * 2 * 2 * 2 * 2
EXPECTED_OUTPUTS_PARALLEL = np.array((np.array(range(100)) * 2).tolist() * 4)


class MultiplyBy(BaseStep):
    def __init__(self, multiply_by=1):
        BaseStep.__init__(self)
        self.multiply_by = multiply_by

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        return self

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

    outputs = p.transform(list(range(15)))

    assert np.array_equal(outputs, np.array(range(15)) * 2 * 2 * 2 * 2)


def test_queued_pipeline_with_step():
    p = SequentialQueuedPipeline([
        MultiplyBy(2),
        MultiplyBy(2),
        MultiplyBy(2),
        MultiplyBy(2)
    ], batch_size=10, n_workers_per_step=1, max_size=5)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_step():
    p = SequentialQueuedPipeline([
        ('1', MultiplyBy(2)),
        ('2', MultiplyBy(2)),
        ('3', MultiplyBy(2)),
        ('4', MultiplyBy(2))
    ], batch_size=10, n_workers_per_step=1, max_size=5)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_n_workers_step():
    p = SequentialQueuedPipeline([
        (1, MultiplyBy(2)),
        (1, MultiplyBy(2)),
        (1, MultiplyBy(2)),
        (1, MultiplyBy(2))
    ], batch_size=10, max_size=5)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_n_workers_step():
    p = SequentialQueuedPipeline([
        (1, MultiplyBy(2)),
        (1, MultiplyBy(2)),
        (1, MultiplyBy(2)),
        (1, MultiplyBy(2))
    ], batch_size=10, max_size=5)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_max_size():
    p = SequentialQueuedPipeline([
        ('1', 1, 5, MultiplyBy(2)),
        ('2', 1, 5, MultiplyBy(2)),
        ('3', 1, 5, MultiplyBy(2)),
        ('4', 1, 5, MultiplyBy(2))
    ], batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_size():
    p = SequentialQueuedPipeline([
        ('1', 1, MultiplyBy(2)),
        ('2', 1, MultiplyBy(2)),
        ('3', 1, MultiplyBy(2)),
        ('4', 1, MultiplyBy(2))
    ], max_size=10, batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_size():
    p = SequentialQueuedPipeline([
        ('1', MultiplyBy(2)),
        ('2', MultiplyBy(2)),
        ('3', MultiplyBy(2)),
        ('4', MultiplyBy(2))
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_parallel_queued_pipeline_with_step_name_n_worker_max_size():
    p = ParallelQueuedFeatureUnion([
        ('1', 1, 5, MultiplyBy(2)),
        ('2', 1, 5, MultiplyBy(2)),
        ('3', 1, 5, MultiplyBy(2)),
        ('4', 1, 5, MultiplyBy(2))
    ], batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_parallelize_correctly():
    sleep_time = 0.001
    p = SequentialQueuedPipeline([
        ('1', 4, 10, Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)])),
        ('2', 4, 10, Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)])),
        ('3', 4, 10, Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)])),
        ('4', 4, 10, Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]))
    ], batch_size=10)

    a = time.time()
    outputs_streaming = p.transform(list(range(100)))
    b = time.time()
    time_queued_pipeline = b - a

    p = Pipeline([
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)])
    ])

    a = time.time()
    outputs_vanilla = p.transform(list(range(100)))
    b = time.time()
    time_vanilla_pipeline = b - a

    assert time_queued_pipeline < time_vanilla_pipeline
    assert np.array_equal(outputs_streaming, outputs_vanilla)


def test_parallel_queued_parallelize_correctly():
    sleep_time = 0.001
    p = SequentialQueuedPipeline([
        ('1', 4, 10, Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)])),
        ('2', 4, 10, Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)])),
        ('3', 4, 10, Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)])),
        ('4', 4, 10, Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]))
    ], batch_size=10)

    a = time.time()
    outputs_streaming = p.transform(list(range(100)))
    b = time.time()
    time_queued_pipeline = b - a

    p = Pipeline([
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)]),
        Pipeline([ForEachDataInput(Sleep(sleep_time=sleep_time)), MultiplyBy(2)])
    ])

    a = time.time()
    outputs_vanilla = p.transform(list(range(100)))
    b = time.time()
    time_vanilla_pipeline = b - a

    assert time_queued_pipeline < time_vanilla_pipeline
    assert np.array_equal(outputs_streaming, outputs_vanilla)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments_max_size():
    n_workers = 4
    worker_arguments = [('multiply_by', 2) for _ in range(n_workers)]
    p = ParallelQueuedFeatureUnion([
        ('1', n_workers, worker_arguments, 5, MultiplyBy()),
    ], batch_size=10)

    outputs = p.transform(list(range(100)))

    expected = np.array(list(range(0, 200, 2)))
    assert np.array_equal(outputs, expected)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments():
    n_workers = 4
    worker_arguments = [('multiply_by', 2) for _ in range(n_workers)]
    p = ParallelQueuedFeatureUnion([
        ('1', n_workers, worker_arguments, MultiplyBy()),
    ], batch_size=10, max_size=5)

    outputs = p.transform(list(range(100)))

    expected = np.array(list(range(0, 200, 2)))
    assert np.array_equal(outputs, expected)


def test_parallel_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_size():
    p = ParallelQueuedFeatureUnion([
        ('1', 1, MultiplyBy(2)),
        ('2', 1, MultiplyBy(2)),
        ('3', 1, MultiplyBy(2)),
        ('4', 1, MultiplyBy(2)),
    ], max_size=10, batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_size():
    p = ParallelQueuedFeatureUnion([
        ('1', MultiplyBy(2)),
        ('2', MultiplyBy(2)),
        ('3', MultiplyBy(2)),
        ('4', MultiplyBy(2)),
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_step_name_n_worker_with_default_n_workers_and_default_max_size():
    p = ParallelQueuedFeatureUnion([
        ('1', MultiplyBy(2)),
        ('2', MultiplyBy(2)),
        ('3', MultiplyBy(2)),
        ('4', MultiplyBy(2)),
    ], n_workers_per_step=1, max_size=10, batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_queued_pipeline_saving(tmpdir):
    # Given
    p = ParallelQueuedFeatureUnion([
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


class QueueJoinerForTest(QueueJoiner):
    def __init__(self, batch_size):
        super().__init__(batch_size)
        self.called_queue_joiner = False

    def join(self, original_data_container: DataContainer) -> DataContainer:
        self.called_queue_joiner = True
        super().join(original_data_container)


def test_sequential_queued_pipeline_should_fit_without_multiprocessing():
    batch_size = 10
    p = SequentialQueuedPipeline([
        (1, FitTransformCallbackStep()),
        (1, FitTransformCallbackStep()),
        (1, FitTransformCallbackStep()),
        (1, FitTransformCallbackStep())
    ], batch_size=batch_size, max_size=5)
    queue_joiner_for_test = QueueJoinerForTest(batch_size=batch_size)
    p.steps[-1] = queue_joiner_for_test
    p.steps_as_tuple[-1] = (p.steps_as_tuple[-1][0], queue_joiner_for_test)
    p._refresh_steps()

    p = p.fit(list(range(100)), list(range(100)))

    assert not p[-1].called_queue_joiner


def test_sequential_queued_pipeline_should_fit_transform_without_multiprocessing():
    batch_size = 10
    p = SequentialQueuedPipeline([
        (1, FitTransformCallbackStep(transform_function=lambda di: np.array(di) * 2)),
        (1, FitTransformCallbackStep(transform_function=lambda di: np.array(di) * 2)),
        (1, FitTransformCallbackStep(transform_function=lambda di: np.array(di) * 2)),
        (1, FitTransformCallbackStep(transform_function=lambda di: np.array(di) * 2))
    ], batch_size=batch_size, max_size=5)
    queue_joiner_for_test = QueueJoinerForTest(batch_size=batch_size)
    p.steps[-1] = queue_joiner_for_test
    p.steps_as_tuple[-1] = (p.steps_as_tuple[-1][0], queue_joiner_for_test)
    p._refresh_steps()

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert not p[-1].called_queue_joiner
    assert np.array_equal(outputs, EXPECTED_OUTPUTS)
