import time

import numpy as np
import pytest

from neuraxle.base import ExecutionContext
from neuraxle.data_container import DataContainer, AbsentValuesNullObject
from neuraxle.distributed.streaming import SequentialQueuedPipeline, ParallelQueuedFeatureUnion, QueueJoiner
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import ForEach
from neuraxle.steps.misc import FitTransformCallbackStep, Sleep
from neuraxle.steps.numpy import MultiplyByN

EXPECTED_OUTPUTS = np.array(range(100)) * 2 * 2 * 2 * 2
EXPECTED_OUTPUTS_PARALLEL = np.array((np.array(range(100)) * 2).tolist() * 4)


def test_queued_pipeline_with_excluded_incomplete_batch():
    p = SequentialQueuedPipeline([
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2)
    ], batch_size=10, keep_incomplete_batch=False, n_workers_per_step=1, max_queue_size=5)

    outputs = p.transform(list(range(15)))

    assert np.array_equal(outputs, np.array(list(range(10))) * 2 * 2 * 2 * 2)


def test_queued_pipeline_with_included_incomplete_batch():
    p = SequentialQueuedPipeline(
        [
            MultiplyByN(2),
            MultiplyByN(2),
            MultiplyByN(2),
            MultiplyByN(2)
        ],
        batch_size=10,
        keep_incomplete_batch=True,
        default_value_data_inputs=AbsentValuesNullObject(),
        default_value_expected_outputs=AbsentValuesNullObject(),
        n_workers_per_step=1,
        max_queue_size=5
    )

    outputs = p.transform(list(range(15)))

    assert np.array_equal(outputs, np.array(list(range(15))) * 2 * 2 * 2 * 2)


def test_queued_pipeline_with_included_incomplete_batch_that_raises_an_exception():
    with pytest.raises(AttributeError):
        p = SequentialQueuedPipeline(
            [
                MultiplyByN(2),
                MultiplyByN(2),
                MultiplyByN(2),
                MultiplyByN(2)
            ],
            batch_size=10,
            keep_incomplete_batch=True,
            default_value_data_inputs=None,  # this will raise an exception in the worker
            default_value_expected_outputs=None,  # this will raise an exception in the worker
            n_workers_per_step=1,
            max_queue_size=5
        )
        p.transform(list(range(15)))


def test_queued_pipeline_with_step_with_process():
    p = SequentialQueuedPipeline([
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2)
    ], batch_size=10, n_workers_per_step=1, max_queue_size=5, use_processes=True)

    data_container = DataContainer(data_inputs=list(range(100)))
    context = ExecutionContext()

    outputs = p.handle_transform(data_container, context)

    assert np.array_equal(outputs.data_inputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_with_threading():
    p = SequentialQueuedPipeline([
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2)
    ], batch_size=10, n_workers_per_step=1, max_queue_size=5, use_processes=False)

    data_container = DataContainer(data_inputs=list(range(100)))
    context = ExecutionContext()

    outputs = p.handle_transform(data_container, context)

    assert np.array_equal(outputs.data_inputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_step():
    p = SequentialQueuedPipeline([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(2)),
        ('3', MultiplyByN(2)),
        ('4', MultiplyByN(2))
    ], batch_size=10, n_workers_per_step=1, max_queue_size=5)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_n_workers_step():
    p = SequentialQueuedPipeline([
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2))
    ], batch_size=10, max_queue_size=5)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_wrapped_queued_pipeline_with_n_workers_step():
    p = Pipeline([SequentialQueuedPipeline([
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2))
    ], batch_size=10, max_queue_size=5)])

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_max_queue_size():
    p = SequentialQueuedPipeline([
        ('1', 1, 5, MultiplyByN(2)),
        ('2', 1, 5, MultiplyByN(2)),
        ('3', 1, 5, MultiplyByN(2)),
        ('4', 1, 5, MultiplyByN(2))
    ], batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_queue_size():
    p = SequentialQueuedPipeline([
        ('1', 1, MultiplyByN(2)),
        ('2', 1, MultiplyByN(2)),
        ('3', 1, MultiplyByN(2)),
        ('4', 1, MultiplyByN(2))
    ], max_queue_size=10, batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_queue_size():
    p = SequentialQueuedPipeline([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(2)),
        ('3', MultiplyByN(2)),
        ('4', MultiplyByN(2))
    ], n_workers_per_step=1, max_queue_size=10, batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS)


def test_parallel_queued_pipeline_with_step_name_n_worker_max_queue_size():
    p = ParallelQueuedFeatureUnion([
        ('1', 1, 5, MultiplyByN(2)),
        ('2', 1, 5, MultiplyByN(2)),
        ('3', 1, 5, MultiplyByN(2)),
        ('4', 1, 5, MultiplyByN(2))
    ], batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


@pytest.mark.parametrize("use_processes", [False, True])
@pytest.mark.parametrize("use_savers", [False, True])
def test_parallel_queued_parallelize_correctly(tmpdir, use_processes, use_savers):
    sleep_time = 0.01
    p = SequentialQueuedPipeline([
        ('1', 2, 10, Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)])),
        ('2', 2, 10, Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)])),
        ('3', 2, 10, Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)])),
        ('4', 2, 10, Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)]))
    ], batch_size=10, use_processes=use_processes, use_savers=use_savers).with_context(ExecutionContext(tmpdir))

    a = time.time()
    outputs_streaming = p.transform(list(range(100)))
    b = time.time()
    time_queued_pipeline = b - a

    p = Pipeline([
        Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)]),
        Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)]),
        Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)]),
        Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)])
    ])

    a = time.time()
    outputs_vanilla = p.transform(list(range(100)))
    b = time.time()
    time_vanilla_pipeline = b - a

    assert time_queued_pipeline < time_vanilla_pipeline
    assert np.array_equal(outputs_streaming, outputs_vanilla)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments_max_queue_size():
    n_workers = 4
    worker_arguments = [('hyperparams', HyperparameterSamples({'multiply_by': 2})) for _ in range(n_workers)]
    p = ParallelQueuedFeatureUnion([
        ('1', n_workers, worker_arguments, 5, MultiplyByN()),
    ], batch_size=10)

    outputs = p.transform(list(range(100)))

    expected = np.array(list(range(0, 200, 2)))
    assert np.array_equal(outputs, expected)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments():
    n_workers = 4
    worker_arguments = [('hyperparams', HyperparameterSamples({'multiply_by': 2})) for _ in range(n_workers)]
    p = ParallelQueuedFeatureUnion([
        ('1', n_workers, worker_arguments, MultiplyByN()),
    ], batch_size=10, max_queue_size=5)

    outputs = p.transform(list(range(100)))

    expected = np.array(list(range(0, 200, 2)))
    assert np.array_equal(outputs, expected)


def test_parallel_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_queue_size():
    p = ParallelQueuedFeatureUnion([
        ('1', 1, MultiplyByN(2)),
        ('2', 1, MultiplyByN(2)),
        ('3', 1, MultiplyByN(2)),
        ('4', 1, MultiplyByN(2)),
    ], max_queue_size=10, batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_queue_size():
    p = ParallelQueuedFeatureUnion([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(2)),
        ('3', MultiplyByN(2)),
        ('4', MultiplyByN(2)),
    ], n_workers_per_step=1, max_queue_size=10, batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


def test_parallel_queued_pipeline_step_name_n_worker_with_default_n_workers_and_default_max_queue_size():
    p = ParallelQueuedFeatureUnion([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(2)),
        ('3', MultiplyByN(2)),
        ('4', MultiplyByN(2)),
    ], n_workers_per_step=1, max_queue_size=10, batch_size=10)

    outputs = p.transform(list(range(100)))

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


@pytest.mark.parametrize("use_processes", [False, True])
@pytest.mark.parametrize("use_savers", [False, True])
def test_queued_pipeline_saving(tmpdir, use_processes, use_savers):
    # Given
    p = ParallelQueuedFeatureUnion([
        ('1', 4, 10, FitTransformCallbackStep()),
        ('2', 4, 10, FitTransformCallbackStep()),
        ('3', 4, 10, FitTransformCallbackStep()),
        ('4', 4, 10, FitTransformCallbackStep()),
    ], n_workers_per_step=4, max_queue_size=10, batch_size=10,
        use_processes=use_processes, use_savers=use_savers).with_context(ExecutionContext(tmpdir))

    # When
    p, _ = p.fit_transform(list(range(200)), list(range(200)))
    p = p.wrapped  # clear execution context wrapper
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

    assert len(p[0].wrapped.transform_callback_function.data) == 20
    assert len(p[0].wrapped.fit_callback_function.data) == 20
    assert len(p[1].wrapped.transform_callback_function.data) == 20
    assert len(p[1].wrapped.fit_callback_function.data) == 20
    assert len(p[2].wrapped.transform_callback_function.data) == 20
    assert len(p[2].wrapped.fit_callback_function.data) == 20
    assert len(p[3].wrapped.transform_callback_function.data) == 20
    assert len(p[3].wrapped.fit_callback_function.data) == 20


def test_queued_pipeline_with_savers(tmpdir):
    # Given
    p = ParallelQueuedFeatureUnion([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(2)),
        ('3', MultiplyByN(2)),
        ('4', MultiplyByN(2)),
    ], n_workers_per_step=1, max_queue_size=10, batch_size=10, use_savers=True).with_context(ExecutionContext(tmpdir))

    # When
    outputs = p.transform(list(range(100)))

    # Then
    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PARALLEL)


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
    ], batch_size=batch_size, max_queue_size=5)
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
    ], batch_size=batch_size, max_queue_size=5, use_processes=False, use_savers=False)
    queue_joiner_for_test = QueueJoinerForTest(batch_size=batch_size)
    p.steps[-1] = queue_joiner_for_test
    p.steps_as_tuple[-1] = (p.steps_as_tuple[-1][0], queue_joiner_for_test)
    p._refresh_steps()

    p, outputs = p.fit_transform(list(range(100)), list(range(100)))

    assert not p[-1].called_queue_joiner
    assert np.array_equal(outputs, EXPECTED_OUTPUTS)
