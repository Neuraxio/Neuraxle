import copy
import random
import time
from pickle import PickleError
from typing import List, Type

import numpy as np
import pytest
from neuraxle.base import BaseStep
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import NonFittableMixin
from neuraxle.data_container import (ARG_X_INPUTTED, ARG_Y_PREDICTD, DACT,
                                     PredsDACT, StripAbsentValues, TrainDACT)
from neuraxle.distributed.streaming import (BaseQueuedPipeline,
                                            ParallelQueuedFeatureUnion,
                                            ParallelWorkersWrapper,
                                            QueuedMinibatchTask,
                                            SequentialQueuedPipeline,
                                            WorkersJoiner)
from neuraxle.hyperparams.space import HyperparameterSamples
from neuraxle.pipeline import Pipeline
from neuraxle.steps.loop import ForEach
from neuraxle.steps.misc import (FitTransformCallbackStep, Sleep,
                                 TransformOnlyCounterLoggingStep)
from neuraxle.steps.numpy import MultiplyByN

GIVEN_BATCH_SIZE = 10
GIVEN_INPUTS: List[int] = list(range(100))
EXPECTED_OUTPUTS_PIPELINE: List[int] = MultiplyByN(2**3).transform(GIVEN_INPUTS).tolist()
EXPECTED_OUTPUTS_PIPELINE_235 = MultiplyByN(2 * 3 * 5).transform(GIVEN_INPUTS).tolist()
EXPECTED_OUTPUTS_FEATURE_UNION: List[int] = (
    MultiplyByN(2).transform(
        GIVEN_INPUTS).tolist() + MultiplyByN(3).transform(
            GIVEN_INPUTS).tolist() + MultiplyByN(5).transform(
                GIVEN_INPUTS).tolist()
)


def test_queued_pipeline_with_excluded_incomplete_batch():
    p = SequentialQueuedPipeline([
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2),
    ], batch_size=GIVEN_BATCH_SIZE, keep_incomplete_batch=False, n_workers_per_step=1, max_queued_minibatches=5)

    outputs = p.transform(list(range(15)))

    assert np.array_equal(outputs, np.array(list(range(10))) * 2**3)


def test_queued_pipeline_with_included_incomplete_batch():
    p = SequentialQueuedPipeline(
        [
            MultiplyByN(2),
            MultiplyByN(2),
            MultiplyByN(2),
        ],
        batch_size=GIVEN_BATCH_SIZE,
        keep_incomplete_batch=True,
        default_value_data_inputs=StripAbsentValues(),
        default_value_expected_outputs=StripAbsentValues(),
        n_workers_per_step=1,
        max_queued_minibatches=5
    )

    outputs = p.transform(list(range(15)))

    assert np.array_equal(outputs, np.array(list(range(15))) * 2**3)


@pytest.mark.timeout(10)
@pytest.mark.parametrize('use_processes', [False, True])
def test_queued_pipeline_can_report_stack_trace_upon_failure(use_processes: bool):
    # TODO: this test runs infinite, it hangs and never ends.
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
    ], batch_size=GIVEN_BATCH_SIZE, n_workers_per_step=1, max_queued_minibatches=5, use_processes=True)

    data_container = DACT(data_inputs=list(range(100)))
    context = CX()

    outputs = p.handle_transform(data_container, context)

    assert np.array_equal(outputs.data_inputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_step_with_threading():
    p = SequentialQueuedPipeline([
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2),
    ], batch_size=GIVEN_BATCH_SIZE, n_workers_per_step=1, max_queued_minibatches=5, use_processes=False)

    data_container = DACT(data_inputs=list(range(100)))
    context = CX()

    outputs = p.handle_transform(data_container, context)

    assert np.array_equal(outputs.data_inputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_step_name_step():
    p = SequentialQueuedPipeline([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(2)),
        ('3', MultiplyByN(2)),
    ], batch_size=GIVEN_BATCH_SIZE, n_workers_per_step=1, max_queued_minibatches=5)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_n_workers_step():
    p = SequentialQueuedPipeline([
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
    ], batch_size=GIVEN_BATCH_SIZE, max_queued_minibatches=5)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_wrapped_queued_pipeline_with_n_workers_step():
    p = Pipeline([SequentialQueuedPipeline([
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
        (1, MultiplyByN(2)),
    ], batch_size=GIVEN_BATCH_SIZE, max_queued_minibatches=5)])

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_step_name_n_worker_max_queued_minibatches():
    p = SequentialQueuedPipeline([
        ('1', 1, 5, MultiplyByN(2)),
        ('2', 1, 5, MultiplyByN(2)),
        ('3', 1, 5, MultiplyByN(2)),
    ], batch_size=GIVEN_BATCH_SIZE)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_queued_minibatches():
    p = SequentialQueuedPipeline([
        ('1', 1, MultiplyByN(2)),
        ('2', 1, MultiplyByN(2)),
        ('3', 1, MultiplyByN(2)),
    ], max_queued_minibatches=10, batch_size=GIVEN_BATCH_SIZE)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_queued_pipeline_with_step_name_n_worker_with_default_n_workers_and_default_max_queued_minibatches():
    p = SequentialQueuedPipeline([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(2)),
        ('3', MultiplyByN(2)),
    ], n_workers_per_step=1, max_queued_minibatches=10, batch_size=GIVEN_BATCH_SIZE)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_PIPELINE)


def test_parallel_queued_pipeline_with_step_name_n_worker_max_queued_minibatches():
    p = ParallelQueuedFeatureUnion([
        ('1', 1, 5, MultiplyByN(2)),
        ('4', 1, 5, MultiplyByN(3)),
        ('3', 1, 5, MultiplyByN(5)),
    ], batch_size=GIVEN_BATCH_SIZE)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_FEATURE_UNION)


@pytest.mark.parametrize('pipeline_class,eo', [
    (SequentialQueuedPipeline, EXPECTED_OUTPUTS_PIPELINE_235),
    (ParallelQueuedFeatureUnion, EXPECTED_OUTPUTS_FEATURE_UNION),
])
def test_parallel_queued_pipeline_that_might_not_reorder_properly_due_to_named_step_order_and_delay(pipeline_class: Type[BaseQueuedPipeline], eo: List[int]):
    p = pipeline_class([
        ('Step1', 1, 5, MultiplyByN(2)),
        # The sleep may push this step named 'Step9' to the end of the processing queue.
        # The name 'Step9' also doesn't sort alphabetically, so the reconstructed order
        # of the steps is even more tested here.
        ('Step9', 1, 5, Pipeline([MultiplyByN(3), Sleep(0.2)])),
        ('Step2', 1, 5, MultiplyByN(5)),
    ], batch_size=GIVEN_BATCH_SIZE, n_workers_per_step=2)
    ids = copy.deepcopy(GIVEN_INPUTS)
    random.shuffle(ids)

    outputs: DACT = p.handle_transform(DACT(ids=ids, di=GIVEN_INPUTS), CX())

    assert np.array_equal(outputs.ids, ids)
    assert np.array_equal(outputs.di, eo)


def test_parallel_queued_threads_do_parallelize_sleep_correctly():
    sleep_time = 0.001
    data_inputs = range(100)
    expected_outputs = MultiplyByN(2**10).transform(data_inputs).tolist()
    sleepers = [
        Pipeline([
            ForEach(Sleep(sleep_time=sleep_time / 2, add_random_quantity=sleep_time)),
            MultiplyByN(2),
            TransformOnlyCounterLoggingStep().set_name(f"SleeperLogger{i}"),
        ])
        for i in range(10)
    ]

    p = SequentialQueuedPipeline(
        sleepers,
        batch_size=5, n_workers_per_step=4, use_processes=False, use_savers=False
    ).with_context(CX())

    a = time.time()
    outputs_streaming = p.transform(data_inputs)
    b = time.time()
    time_queued_pipeline = b - a

    p = Pipeline(
        sleepers,
    )

    a = time.time()
    outputs_vanilla = p.transform(data_inputs)
    b = time.time()
    time_vanilla_pipeline = b - a

    assert np.array_equal(outputs_vanilla, expected_outputs)
    assert len(outputs_streaming) == len(expected_outputs), (outputs_streaming, expected_outputs)
    assert np.array_equal(outputs_streaming, expected_outputs)
    assert time_queued_pipeline < time_vanilla_pipeline


def test_parallel_queued_pipeline_with_step_name_n_worker_with_step_name_n_workers_and_default_max_queued_minibatches():
    p = ParallelQueuedFeatureUnion([
        ('1', 1, MultiplyByN(2)),
        ('2', 1, MultiplyByN(3)),
        ('3', 1, MultiplyByN(5)),
    ], max_queued_minibatches=10, batch_size=GIVEN_BATCH_SIZE)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_FEATURE_UNION)


@pytest.mark.parametrize('use_processes', [True, False])
def test_parallel_queued_pipeline_with_2_workers_and_small_queue_size(use_processes: bool):
    # TODO: cascading sleeps to make first steps process faster to assert on logs that the next queued steps do wait with restricted queue sizes?
    p = ParallelQueuedFeatureUnion([
        MultiplyByN(2),
        MultiplyByN(3),
        MultiplyByN(5),
    ], max_queued_minibatches=4, batch_size=GIVEN_BATCH_SIZE, use_processes=use_processes, n_workers_per_step=2)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_FEATURE_UNION)


def test_parallel_queued_pipeline_with_workers_and_batch_and_queue_of_ample_size():
    p = ParallelQueuedFeatureUnion([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(3)),
        ('3', MultiplyByN(5)),
    ], n_workers_per_step=1, max_queued_minibatches=10, batch_size=GIVEN_BATCH_SIZE)

    outputs = p.transform(GIVEN_INPUTS)

    assert np.array_equal(outputs, EXPECTED_OUTPUTS_FEATURE_UNION)


@pytest.mark.parametrize("use_processes", [False, True])
@pytest.mark.parametrize("use_savers", [False, True])
def test_queued_pipeline_multiple_workers(use_processes, use_savers):
    cx = CX()
    # Given
    p = ParallelQueuedFeatureUnion([
        FitTransformCallbackStep(),
        FitTransformCallbackStep(),
        FitTransformCallbackStep(),
        FitTransformCallbackStep(),
    ], n_workers_per_step=4, max_queued_minibatches=10, batch_size=GIVEN_BATCH_SIZE,
        use_processes=use_processes, use_savers=use_savers).with_context(cx)

    # When
    p, _ = p.fit_transform(list(range(200)), list(range(200)))
    p = p.wrapped  # clear execution context wrapper
    p.save(cx)
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

    p = p.load(cx)

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
    ], n_workers_per_step=1, max_queued_minibatches=10, batch_size=GIVEN_BATCH_SIZE,
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

    def join_workers(self, original_dact: DACT, sync_context: CX) -> DACT:
        self.called_queue_joiner = True
        raise NotImplementedError("This method should not be called.")
        # super().join_workers(original_dact, sync_context)


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


@pytest.mark.parametrize("use_savers", [False, True])
@pytest.mark.parametrize("use_processes", [False, True])
@pytest.mark.parametrize('pipeline_class', [SequentialQueuedPipeline, ParallelQueuedFeatureUnion])
def test_parallel_logging_works_with_streamed_steps(pipeline_class: BaseQueuedPipeline, use_processes: bool, use_savers: bool):
    # TODO: parametrize using SequentialQueuedPipeline as well instead of ParallelQueuedFeatureUnion
    # Given
    minibatch_size = 50
    n_workers = 2
    names = [f"{i}-{pipeline_class.__name__}-{use_processes}-{use_savers}" for i in range(3)]
    p = pipeline_class([
        TransformOnlyCounterLoggingStep().set_name(names[0]),
        TransformOnlyCounterLoggingStep().set_name(names[1]),
        TransformOnlyCounterLoggingStep().set_name(names[2]),
    ], n_workers_per_step=n_workers, batch_size=minibatch_size,
        use_processes=use_processes, use_savers=use_savers)

    # When
    p.transform(GIVEN_INPUTS)

    # Then
    log_history = list(CX().logger)
    n_calls = int(len(GIVEN_INPUTS) / minibatch_size) * 3
    # NOTE: logs are duplicated by pytest-xdist's handlers or something, and polluted by other parallel tests. See: https://github.com/pytest-dev/pytest/issues/10062
    shortened_log_history = [e[:e.index('#') + 1] for e in set(log_history) if "logging call" in e]
    assert len(shortened_log_history) >= n_calls, str(shortened_log_history)
    for nm in names:
        log_line = f"{nm} - transform call - logging call #"
        assert log_line in shortened_log_history, str(shortened_log_history)
    pass


@pytest.mark.parametrize('batches_count', [1, 4, 8])
@pytest.mark.parametrize('use_processes', [False, True])
def test_parallel_workers_wrapper_for_some_batches(batches_count: int, use_processes: bool):
    n_parallel_workers = 4
    step = TransformOnlyCounterLoggingStep().set_name(f"{batches_count}BatchesLogger")
    cx = CX()
    cx.synchroneous()
    worker = ParallelWorkersWrapper(
        step,
        n_workers=n_parallel_workers,
        use_processes=use_processes,
    )
    worker._setup(context=cx)
    joiner = WorkersJoiner(batch_size=GIVEN_BATCH_SIZE, n_worker_wrappers_to_join=1)
    joiner._setup(context=cx)
    worker.register_consumer(joiner)
    whole_batch_dact = DACT(di=list(range(10 * batches_count)))
    minibatches_dacts = [DACT(di=list(range(i * 10, (i + 1) * 10))) for i in range(batches_count)]
    minibatches_tasks = [QueuedMinibatchTask(minibatch_dact=dact, step_name=step.name) for dact in minibatches_dacts]
    joiner.append_terminal_summary(worker.name, minibatches_tasks[-1])
    worker.start(cx)

    for mbt in minibatches_tasks:
        worker.put_minibatch_produced(mbt)
    joiner.set_join_quantities(1, batches_count)
    _out: DACT = joiner.join_workers(whole_batch_dact, cx)
    worker.join()
    worker.teardown()
    joiner.teardown()

    assert _out == whole_batch_dact
    logs = "\n".join(cx.logger)
    assert f"{step.name} - transform call - logging call" in logs


def test_parallel_workers_wrapper_for_no_batches():
    n_parallel_workers = 4
    step = TransformOnlyCounterLoggingStep().set_name("NoBatchLogger")
    cx = CX()
    cx.synchroneous()
    worker = ParallelWorkersWrapper(
        step,
        n_workers=n_parallel_workers,
        use_processes=False,
    )
    worker._setup(context=cx)
    joiner = WorkersJoiner(batch_size=GIVEN_BATCH_SIZE, n_worker_wrappers_to_join=0)
    joiner._setup(context=cx)
    worker.register_consumer(joiner)
    whole_batch_dact = DACT(di=[])
    worker.start(cx)

    joiner.set_join_quantities(1, 0)
    _out: DACT = joiner.join_workers(whole_batch_dact, cx)
    worker.join()
    worker.teardown()
    joiner.teardown()

    assert _out == whole_batch_dact
    logs = "\n".join(cx.logger)
    assert f"{step.name} - transform call - logging call #" not in logs


@pytest.mark.parametrize("use_savers", [False, True])
@pytest.mark.parametrize("use_processes", [False, True])
@pytest.mark.parametrize('pipeline_class', [SequentialQueuedPipeline, ParallelQueuedFeatureUnion])
def test_can_reuse_streaming_step_with_several_varied_batches(pipeline_class: BaseQueuedPipeline, use_savers: bool, use_processes: bool):
    p = pipeline_class([
        ('1', MultiplyByN(2)),
        ('2', MultiplyByN(3)),
        ('3', MultiplyByN(5)),
    ], n_workers_per_step=2, max_queued_minibatches=10, batch_size=GIVEN_BATCH_SIZE, use_savers=use_savers, use_processes=use_processes)
    given_inputs_1 = GIVEN_INPUTS
    given_inputs_2 = list(range(300, 350))
    if pipeline_class == SequentialQueuedPipeline:
        expected_outputs_1 = EXPECTED_OUTPUTS_PIPELINE_235
        expected_outputs_2 = MultiplyByN(2 * 3 * 5).transform(given_inputs_2).tolist()
    else:
        expected_outputs_1 = EXPECTED_OUTPUTS_FEATURE_UNION
        expected_outputs_2 = MultiplyByN(2).transform(
            given_inputs_2).tolist() + MultiplyByN(3).transform(
                given_inputs_2).tolist() + MultiplyByN(5).transform(
                    given_inputs_2).tolist()

    outputs_1 = p.transform(given_inputs_1)
    outputs_2 = p.transform(given_inputs_2)

    assert np.array_equal(outputs_1, expected_outputs_1)
    assert np.array_equal(outputs_2, expected_outputs_2)


def test_wrapped_queued_pipeline_with_0_workers_still_uses_1_worker():
    p = SequentialQueuedPipeline([
        MultiplyByN(2),
        MultiplyByN(2),
        MultiplyByN(2),
    ], batch_size=GIVEN_BATCH_SIZE, max_queued_minibatches=5, n_workers_per_step=0)

    assert all(1 == b.n_workers for b in p.body)


class UnpicklableContextReturnedAsTransformDact(NonFittableMixin, BaseStep):
    def __init__(self):
        BaseStep.__init__(self)
        NonFittableMixin.__init__(self)

    def _transform_data_container(self, data_container: TrainDACT, context: CX) -> PredsDACT:
        return context

    def transform(self, data_inputs: ARG_X_INPUTTED) -> ARG_Y_PREDICTD:
        cx = CX()
        cx.synchroneous()
        return cx


@pytest.mark.timeout(10)
def test_worker_unpicklable_data():
    unpicklable_cx = CX()
    unpicklable_cx.synchroneous()
    batch_size = 10
    p = SequentialQueuedPipeline([
        UnpicklableContextReturnedAsTransformDact()
    ], batch_size=batch_size, n_workers_per_step=2)

    with pytest.raises(ValueError):
        # If it doesn't raise the PickleError, then the thread will probably wait forever.
        # That is why a timeout needs to be used for this test.
        p.handle_transform(DACT(di=list(range(batch_size))), unpicklable_cx)


def test_services_process_safe_items():
    pass
