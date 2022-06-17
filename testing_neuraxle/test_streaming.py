import copy
import random
import time
from typing import List, Type

import numpy as np
import pytest

from neuraxle.base import ExecutionContext as CX, NonFittableMixin
from neuraxle.data_container import DACT, StripAbsentValues
from neuraxle.distributed.streaming import BaseQueuedPipeline, SequentialQueuedPipeline, ParallelQueuedFeatureUnion, WorkersJoiner
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import ForEach
from neuraxle.steps.misc import FitTransformCallbackStep, Sleep
from neuraxle.steps.numpy import MultiplyByN
from testing_neuraxle.test_context_logger import FitTransformCounterLoggingStep


GIVEN_INPUTS: List[int] = list(range(100))
EXPECTED_OUTPUTS_PIPELINE: List[int] = MultiplyByN(2**3).transform(GIVEN_INPUTS).tolist()
EXPECTED_OUTPUTS_FEATURE_UNION: List[int] = MultiplyByN(2).transform(GIVEN_INPUTS).tolist(
) + MultiplyByN(3).transform(GIVEN_INPUTS).tolist() + MultiplyByN(5).transform(GIVEN_INPUTS).tolist()


def test_queued_pipeline_with_excluded_incomplete_batch():
    p = SequentialQueuedPipeline([
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2),
    ], batch_size=10, keep_incomplete_batch=False, n_workers_per_step=1, max_queued_minibatches=5)

    outputs = p.transform(list(range(15)))

    assert np.array_equal(outputs, np.array(list(range(10))) * 2**3)


def test_queued_pipeline_with_included_incomplete_batch():
    p = SequentialQueuedPipeline(
        [
            MultiplyByN(2),
            MultiplyByN(2),
            MultiplyByN(2),
        ],
        batch_size=10,
        keep_incomplete_batch=True,
        default_value_data_inputs=StripAbsentValues(),
        default_value_expected_outputs=StripAbsentValues(),
        n_workers_per_step=1,
        max_queued_minibatches=5
    )

    outputs = p.transform(list(range(15)))

    assert np.array_equal(outputs, np.array(list(range(15))) * 2**3)


@pytest.mark.parametrize('use_processes', [False, True])
def test_queued_pipeline_can_report_stack_trace_upon_failure(use_processes: bool):
    batch_size = 10
    uneven_total_batch = 11
    with pytest.raises(TypeError):
        p = SequentialQueuedPipeline(
            [
                MultiplyByN(2),
                MultiplyByN(2),
                MultiplyByN(2),
            ],
            batch_size=batch_size,
            keep_incomplete_batch=True,
            # This will raise an exception in the worker because MultiplyByN will not be able to multiply the default value "None":
            default_value_data_inputs=None,
            default_value_expected_outputs=None,
            n_workers_per_step=1,
            max_queued_minibatches=5,
            use_processes=use_processes,
        )

        p.transform(list(range(uneven_total_batch)))

    log_history = CX().logger.get_root_string_history()
    expected_error_logged = "unsupported operand type(s) for *: 'NoneType' and 'int'"
    assert expected_error_logged in log_history
    expected_error_stack_trace_line = "return data_inputs * self.hyperparams['multiply_by']"
    assert expected_error_stack_trace_line in log_history


def test_queued_pipeline_with_step_with_process():
    p = SequentialQueuedPipeline([
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2),
    ], batch_size=10, n_workers_per_step=1, max_queued_minibatches=5, use_processes=True)

    data_container = DACT(data_inputs=list(range(100)))
    context = CX()

    outputs = p.handle_transform(data_container, context)

    assert np.array_equal(outputs.data_inputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_step_with_threading():
    p = SequentialQueuedPipeline([
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2),
    ], batch_size=10, n_workers_per_step=1, max_queued_minibatches=5, use_processes=False)

    data_container = DACT(data_inputs=list(range(100)))
    context = CX()

    outputs = p.handle_transform(data_container, context)

    assert np.array_equal(outputs.data_inputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_step_name_step():
    p = SequentialQueuedPipeline([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(2)),
        ('3', MultiplyByN(2)),
    ], batch_size=10, n_workers_per_step=1, max_queued_minibatches=5)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_n_workers_step():
    p = SequentialQueuedPipeline([
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
    ], batch_size=10, max_queued_minibatches=5)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_wrapped_queued_pipeline_with_n_workers_step():
    p = Pipeline([SequentialQueuedPipeline([
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
    ], batch_size=10, max_queued_minibatches=5)])

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_step_name_n_worker_max_queued_minibatches():
    p = SequentialQueuedPipeline([
        ('1', 1, 5, MultiplyByN(2)),
        ('2', 1, 5, MultiplyByN(2)),
        ('3', 1, 5, MultiplyByN(2)),
    ], batch_size=10)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_queued_minibatches():
    p = SequentialQueuedPipeline([
        ('1', 1, MultiplyByN(2)),
        ('2', 1, MultiplyByN(2)),
        ('3', 1, MultiplyByN(2)),
    ], max_queued_minibatches=10, batch_size=10)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_queued_minibatches():
    p = SequentialQueuedPipeline([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(2)),
        ('3', MultiplyByN(2)),
    ], n_workers_per_step=1, max_queued_minibatches=10, batch_size=10)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_parallel_queued_pipeline_with_step_name_n_worker_max_queued_minibatches():
    p = ParallelQueuedFeatureUnion([
        ('1', 1, 5, MultiplyByN(2)),
        ('4', 1, 5, MultiplyByN(3)),
        ('3', 1, 5, MultiplyByN(5)),
    ], batch_size=10)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_FEATURE_UNION)


# parametrize on SequentialQueuedPipeline and ParallelQueuedFeatureUnion:
@pytest.mark.parametrize('pipeline_class,eo', [
    (SequentialQueuedPipeline, MultiplyByN(2 * 3 * 5).transform(GIVEN_INPUTS).tolist()),
    (ParallelQueuedFeatureUnion, EXPECTED_OUTPUTS_FEATURE_UNION),
])
def test_parallel_queued_pipeline_that_might_not_reorder_properly_due_to_named_step_order_and_delay(pipeline_class: Type[BaseQueuedPipeline], eo: List[int]):
    # TODO: do same test for queued pipeline or parametrize test.
    p = pipeline_class([
        ('Step1', 1, 5, MultiplyByN(2)),
        # The sleep may push this step named 'Step9' to the end of the processing queue.
        # The name 'Step9' also doesn't sort alphabetically, so the reconstructed order
        # of the steps is even more tested here.
        ('Step9', 1, 5, Pipeline([MultiplyByN(3), Sleep(0.2)])),
        ('Step2', 1, 5, MultiplyByN(5)),
    ], batch_size=10, n_workers_per_step=2)
    ids = copy.deepcopy(GIVEN_INPUTS)
    random.shuffle(ids)

    outputs: DACT = p.handle_transform(DACT(ids=ids, di=GIVEN_INPUTS), CX())

    assert np.array_equal(outputs.ids, ids)
    assert np.array_equal(outputs.di, eo)


def test_parallel_queued_threads_do_parallelize_sleep_correctly(tmpdir):
    sleep_time = 0.01
    p = SequentialQueuedPipeline([
        ('1', 2, 10, Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)])),
        ('2', 2, 10, Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)])),
        ('3', 2, 10, Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)])),
        ('4', 2, 10, Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)]))
    ], batch_size=10, use_processes=False, use_savers=False).with_context(CX(tmpdir))

    a = time.time()
    outputs_streaming = p.transform(GIVEN_INPUTS)
    b = time.time()
    time_queued_pipeline = b - a

    p = Pipeline([
        Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)]),
        Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)]),
        Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)]),
        Pipeline([ForEach(Sleep(sleep_time=sleep_time)), MultiplyByN(2)])
    ])

    a = time.time()
    outputs_vanilla = p.transform(GIVEN_INPUTS)
    b = time.time()
    time_vanilla_pipeline = b - a

    assert time_queued_pipeline < time_vanilla_pipeline
    assert np.array_equal(outputs_streaming, outputs_vanilla)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments_max_queued_minibatches():
    n_workers = 4
    worker_arguments = [('hyperparams', HyperparameterSamples({'multiply_by': 2})) for _ in range(n_workers)]
    p = ParallelQueuedFeatureUnion([
        ('1', n_workers, worker_arguments, 5, MultiplyByN()),
    ], batch_size=10)

    outputs = p.transform(GIVEN_INPUTS)

    expected = np.array(list(range(0, 200, 2)))
    assert np.array_equal(outputs, expected)


def test_parallel_queued_pipeline_with_step_name_n_worker_additional_arguments():
    n_workers = 4
    worker_arguments = [('hyperparams', HyperparameterSamples({'multiply_by': 2})) for _ in range(n_workers)]
    p = ParallelQueuedFeatureUnion([
        ('1', n_workers, worker_arguments, MultiplyByN()),
    ], batch_size=10, max_queued_minibatches=5)

    outputs = p.transform(GIVEN_INPUTS)

    expected = np.array(list(range(0, 200, 2)))
    assert np.array_equal(outputs, expected)


def test_parallel_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_queued_minibatches():
    p = ParallelQueuedFeatureUnion([
        ('1', 1, MultiplyByN(2)),
        ('2', 1, MultiplyByN(3)),
        ('3', 1, MultiplyByN(5)),
    ], max_queued_minibatches=10, batch_size=10)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_FEATURE_UNION)


@pytest.mark.parametrize('use_processes', [True, False])
def test_parallel_queued_pipeline_with_2_workers_and_small_queue_size(use_processes: bool):
    # TODO: cascading sleeps to make first steps process faster.
    p = ParallelQueuedFeatureUnion([
        MultiplyByN(2),
        MultiplyByN(3),
        MultiplyByN(5),
    ], max_queued_minibatches=4, batch_size=10, use_processes=use_processes, n_workers_per_step=2)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_FEATURE_UNION)


def test_parallel_queued_pipeline_with_workers_and_batch_and_queue_of_ample_size():
    p = ParallelQueuedFeatureUnion([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(3)),
        ('3', MultiplyByN(5)),
    ], n_workers_per_step=1, max_queued_minibatches=10, batch_size=10)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_FEATURE_UNION)


@pytest.mark.parametrize("use_processes", [False, True])
@pytest.mark.parametrize("use_savers", [False, True])
def test_queued_pipeline_multiple_workers(tmpdir, use_processes, use_savers):
    # Given
    p = ParallelQueuedFeatureUnion([
        FitTransformCallbackStep(),
        FitTransformCallbackStep(),
        FitTransformCallbackStep(),
        FitTransformCallbackStep(),
    ], n_workers_per_step=4, max_queued_minibatches=10, batch_size=10,
        use_processes=use_processes, use_savers=use_savers).with_context(CX(tmpdir))

    # When
    p, _ = p.fit_transform(list(range(200)), list(range(200)))
    p = p.wrapped  # clear execution context wrapper
    p.save(CX(tmpdir))
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

    p = p.load(CX(tmpdir))

    assert len(p[0].wrapped.transform_callback_function.data) == 20
    assert len(p[0].wrapped.fit_callback_function.data) == 20
    assert len(p[1].wrapped.transform_callback_function.data) == 20
    assert len(p[1].wrapped.fit_callback_function.data) == 20
    assert len(p[2].wrapped.transform_callback_function.data) == 20
    assert len(p[2].wrapped.fit_callback_function.data) == 20
    assert len(p[3].wrapped.transform_callback_function.data) == 20
    assert len(p[3].wrapped.fit_callback_function.data) == 20


@pytest.mark.parametrize("use_processes", [False, True])
def test_queued_pipeline_with_savers(tmpdir, use_processes: bool):
    # Given
    context = CX()
    p = ParallelQueuedFeatureUnion([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(3)),
        ('3', MultiplyByN(5)),
    ], n_workers_per_step=1, max_queued_minibatches=10, batch_size=10,
        use_savers=True, use_processes=use_processes
    ).with_context(context)

    # When
    outputs = p.transform(GIVEN_INPUTS)

    # Then
    assert np.array_equal(outputs, EXPECTED_OUTPUTS_FEATURE_UNION)


class QueueJoinerForTest(WorkersJoiner):
    def __init__(self, batch_size):
        super().__init__(batch_size)
        self.called_queue_joiner = False

    def join(self, original_data_container: DACT) -> DACT:
        self.called_queue_joiner = True
        super().join(original_data_container)


def test_sequential_queued_pipeline_should_fit_without_multiprocessing():
    batch_size = 10
    p = SequentialQueuedPipeline([
        (1, FitTransformCallbackStep()),
        (1, FitTransformCallbackStep()),
        (1, FitTransformCallbackStep()),
        (1, FitTransformCallbackStep())
    ], batch_size=batch_size, max_queued_minibatches=5)
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
    ], batch_size=batch_size, max_queued_minibatches=5, use_processes=False, use_savers=False)
    queue_joiner_for_test = QueueJoinerForTest(batch_size=batch_size)
    p.steps[-1] = queue_joiner_for_test
    p.steps_as_tuple[-1] = (p.steps_as_tuple[-1][0], queue_joiner_for_test)
    p._refresh_steps()

    p, outputs = p.fit_transform(GIVEN_INPUTS, GIVEN_INPUTS)

    assert not p[-1].called_queue_joiner
    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


class TransformOnlyCounterLoggingStep(NonFittableMixin, FitTransformCounterLoggingStep):
    def __init__(self):
        FitTransformCounterLoggingStep.__init__(self)
        NonFittableMixin.__init__(self)


@pytest.mark.parametrize("use_processes", [False, True])
def test_parallel_logging_works_with_streamed_steps(use_processes: bool):
    # TODO: parametrize using SequentialQueuedPipeline as well instead of ParallelQueuedFeatureUnion
    # Given
    minibatch_size = 50
    n_workers = 2
    p = ParallelQueuedFeatureUnion([
        TransformOnlyCounterLoggingStep().set_name('1'),
        TransformOnlyCounterLoggingStep().set_name('2'),
        TransformOnlyCounterLoggingStep().set_name('3'),
    ], n_workers_per_step=n_workers, batch_size=minibatch_size,
        use_processes=use_processes, use_savers=False)

    # When
    p.transform(GIVEN_INPUTS)

    # Then
    log_history = list(CX().logger)
    n_calls = int(len(GIVEN_INPUTS) / minibatch_size) * 3
    assert len(log_history) == n_calls, log_history
    for name in ['1', '2', '3']:
        log_line0 = f"{name} - transform call - logging call #0"
        log_line1 = f"{name} - transform call - logging call #1"
        assert log_line0 in log_history or log_line1 in log_history, log_history
