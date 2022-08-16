"""
Streaming Pipelines for Parallel and Queued Data Processing
===================================================================

Neuraxle steps for streaming data in parallel in the pipeline.

Pipelines can stream data in queues with workers for each steps.
Max queue sizes can be set, as well as number of clones per steps
for the transformers.


..
    Copyright 2022, Neuraxio Inc.

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

"""
import pickle
import queue
import traceback
from abc import abstractmethod
from collections import OrderedDict, defaultdict
from multiprocessing import Lock, Process, Queue, RLock
from multiprocessing.dummy import current_process
from threading import Thread, current_thread
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from neuraxle.base import BaseSaver, BaseTransformer
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import MetaStep, MixinForBaseTransformer, NamedStepsList, NonFittableMixin, _FittableStep
from neuraxle.data_container import DACT, DIT, EOT, IDT, ListDataContainer, PredsDACT, StripAbsentValues
from neuraxle.hyperparams.space import RecursiveDict
from neuraxle.logging.logging import (ParallelLoggingConsumerThread,
                                      register_log_producer_for_main_logger_thread_to_consume)
from neuraxle.pipeline import Joiner, MiniBatchSequentialPipeline, Pipeline
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch


class _ProducerConsumerStepSaver(BaseSaver):
    """
    Saver for :class:`_ProducerConsumerMixin`.
    This saver class makes sure that the non-picklable queue
    is deleted upon saving for multiprocessing steps.
    """

    def save_step(self, step: BaseTransformer, context: 'CX') -> BaseTransformer:
        step: _ProducerConsumerMixin = step  # typing.
        step._allow_exit_without_queue_flush()
        step.input_queue = None
        step.consumers = []
        return step

    def can_load(self, step: BaseTransformer, context: 'CX') -> bool:
        return True

    def load_step(self, step: 'BaseTransformer', context: 'CX') -> 'BaseTransformer':
        step: _ProducerConsumerMixin = step  # typing.
        step.input_queue = None
        return step


class _QueueDestroyedError(EOFError):
    """
    Error raised when the queue is destroyed.
    """
    pass


EMPTY_CONSUMER_QUEUE_TIMEOUT_SECS = 0.1


class _ProducerConsumerMixin(MixinForBaseTransformer):
    """
    A class to represent a step that can receive minibatches from a producer in its queue to consume them.
    Once minibatches are consumed by the present step, they are produced back to the next consumers in line.

    The Queue in self is the one at the entry to be consumed by self. The output queue is external, in the registred consumers.

    Therefore, in this sense, consumers and producers are both themselves instances of _HasQueueMixin and play the two roles unless they are the first an lasts of their multiprocessing pipelines.
    They will thus be used to consume the produced tasks added to them in their other threads or processes.
    """

    def __init__(self, max_queued_minibatches: int = 0):
        # TODO: two class for this: a producer class and a consumer class perhaps?
        MixinForBaseTransformer.__init__(self)
        self.savers.append(_ProducerConsumerStepSaver())

        self.max_queued_minibatches: int = max(0, max_queued_minibatches or 0)

        self.input_queue: Queue = None
        self.consumers: List[Queue] = []

    def _setup(self, context: 'CX' = None) -> Optional[RecursiveDict]:
        self._init_queue()

    def _init_queue(self):
        if self.input_queue is None:
            self.input_queue = Queue(maxsize=self.max_queued_minibatches)

    def _teardown(self) -> Optional[RecursiveDict]:
        self.join()
        self.input_queue = None
        return RecursiveDict()

    def register_consumer(self, recepient: '_ProducerConsumerMixin') -> '_ProducerConsumerMixin':
        """
        Add a consumer to self.consumers so that when self produces a minibatch, it allows consumers to consume it.
        """
        self._init_queue()
        recepient._init_queue()
        self.consumers.append(recepient.input_queue)
        return self

    def put_minibatch_produced_to_next_consumers(self, task: 'QueuedMinibatchTask'):
        """
        Push a minibatch to all subsequent consumers to allow them to consume it. If the task is terminal, close our own queue.
        """
        task.worker_name = self.name
        self._ensure_task_picklable(task)
        for consumers_of_self in self.consumers:
            consumers_of_self: _ProducerConsumerMixin  # typing...
            consumers_of_self.put(task)

    def put_minibatch_produced(self, task: 'QueuedMinibatchTask'):
        """
        Put a minibatch in queue.
        The caller of this method is the producer of the minibatch and is external to self.
        """
        self._ensure_task_picklable(task)
        try:
            self.input_queue.put(task)
            if task.is_error():
                self._allow_exit_without_queue_flush()
        except Exception as e:
            raise _QueueDestroyedError("It seems like the queue to produce to has been destroyed.") from e

    def _ensure_task_picklable(self, task: 'QueuedMinibatchTask'):
        try:
            pickle.dumps(task)
        except Exception as e:
            # See: https://github.com/python/cpython/issues/79423
            raise pickle.PickleError(f"Couldn't pickle task {task} to send it to the next consumers.") from e

    def _get_minibatch_to_consume(self) -> 'QueuedMinibatchTask':
        """
        Get last minibatch in queue. The caller of this method is probably self,
        that is why the method is private (starts with an underscore).
        This method can raise an EOFError.
        """
        try:
            task: QueuedMinibatchTask = self.input_queue.get(block=True, timeout=EMPTY_CONSUMER_QUEUE_TIMEOUT_SECS)
            if task.is_error():
                self._allow_exit_without_queue_flush()
            return task
        except queue.Empty as e:
            # only happens when block=False or with a timeout:
            raise e from e
        except Exception as e:
            raise _QueueDestroyedError("It seems like the queue to consume from has been destroyed.") from e

    def join(self):
        if self.input_queue is not None:
            self.input_queue.close()
            self._allow_exit_without_queue_flush()
            self.input_queue.join_thread()
            self.input_queue = None

    def _allow_exit_without_queue_flush(self):
        if self.input_queue is not None:
            self.input_queue.cancel_join_thread()


class QueuedMinibatchTask:
    """
    Data object to contain the minibatch processed by producers and consumers.
    """

    def __init__(self, minibatch_dact: DACT, step_name: str = None):
        self.worker_name: str = step_name
        self.minibatch_dact: ListDataContainer = minibatch_dact

    def is_error(self) -> bool:
        return False

    def to_error(self, error: Exception) -> 'MinibatchError':
        _tb_str: str = traceback.format_exc()
        _thread_name: str = current_thread().name
        _process_name: str = current_process().name
        traceback_msg = f'{error}\n\nProcess:Thread "{_process_name}:{_thread_name}" {_tb_str}'

        return MinibatchError(
            minibatch_dact=self.minibatch_dact,
            step_name=self.worker_name,
            err=error,
            traceback_msg=traceback_msg)


class MinibatchError(QueuedMinibatchTask):
    """
    Data object to represent an error in a minibatch.
    """

    def __init__(self, minibatch_dact: DACT, step_name: str, err: Exception, traceback_msg=None):
        super().__init__(minibatch_dact, step_name)
        self.err: Exception = err
        self.traceback_msg: str = traceback_msg

    def is_error(self) -> bool:
        return True

    def get_err(self) -> Exception:
        if self.traceback_msg is not None:
            return type(self.err)(self.traceback_msg)
        return self.err


def worker_function(
    worker: 'ParallelWorkersWrapper',
    context: CX,
    use_savers: bool,
    logging_queue: Optional[Queue],
):
    """
    Worker function that transforms the items inside the queue of items to process.

    :param queue_worker: step to transform
    :param context: execution context
    :param use_savers: use savers
    :return:
    """
    try:
        if use_savers:
            worker.reload_post_saving(context)
        step = worker.get_step()

        register_log_producer_for_main_logger_thread_to_consume(logging_queue)

        while True:
            task: QueuedMinibatchTask = None
            try:
                task = worker._get_minibatch_to_consume()
                if task.is_error():
                    break
                task.minibatch_dact = step.handle_transform(
                    task.minibatch_dact, context)

                if not isinstance(task.minibatch_dact, DACT):
                    raise ValueError(
                        f"Minibatch DACT is not of good type. Received {type( task.minibatch_dact)}: {str( task.minibatch_dact)}.")
                    pickle.dumps(task)

            except queue.Empty:
                # Queue did a timeout, continuing.
                task = None
                continue  # breakpoint here.
            except _QueueDestroyedError:
                # This error can happen at the destruction of workers or processes.
                task = None
                return  # breakpoint here.
            except Exception as err:
                context.flow.log_error(err)
                task.minibatch_dact = None
                task = pickle_exception_into_task(task, err)
                pass
            finally:
                if task is not None:
                    worker.put_minibatch_produced_to_next_consumers(task)
    except Exception as err:
        context.flow.log_error(err)
        task = pickle_exception_into_task(task, err)
        worker.put_minibatch_produced_to_next_consumers(task)
    else:
        pass  # This pass is for breakpoints to be set if needed.
    finally:
        pass  # This pass is for breakpoints to be set if needed.


def pickle_exception_into_task(task, err) -> MinibatchError:
    if task is None:
        task = QueuedMinibatchTask(None, None)
    return task.to_error(err)


class ParallelWorkersWrapper(_ProducerConsumerMixin, MetaStep):
    """
    Start multiple Process or Thread that consumes items from the minibatch DACT Queue, and produces them on the next registered consumers' queue.
    """

    def __init__(
        self,
        wrapped: BaseTransformer,
        max_queued_minibatches: int = None,
        n_workers: int = 1,
        use_processes: bool = True,
        use_savers: bool = False
    ):
        MetaStep.__init__(self, wrapped)
        _ProducerConsumerMixin.__init__(self, max_queued_minibatches)
        self.n_workers: int = n_workers
        self.use_processes: bool = use_processes
        self.use_savers = use_savers

        self.running_workers: List[Process] = []
        # TODO: maybe start this thread in the main instead than in every worker's main.
        self.logging_thread: Optional[ParallelLoggingConsumerThread] = None

    def _setup(self, context: 'CX' = None) -> Optional[RecursiveDict]:
        MetaStep._setup(self)
        _ProducerConsumerMixin._setup(self)

    def __getstate__(self):
        """
        This class, upon being forked() to a new process with pickles,
        should not copy references to other threads or processes.
        """
        state = self.__dict__.copy()
        state['running_workers'] = []  # delete self.workers in the process fork's pickling.
        state['logging_thread'] = None
        return state

    def start(self, context: CX, logging_queue: Optional[Queue] = None):
        """
        Start multiple processes or threads with the worker function as a target.
        These workers will consume minibatches from the queue and produce them on the next queue(s).
        They are started as multiprocessing daemons, so that they will not block the main
        process if there is an error requiring to exit.

        :param context: An execution context that will be checked to be thread_safe.
        :param logging_queue: An optional logging_queue from the object :class:`ParallelLoggingConsumerThread` to pass and recover parallelized log records to. Not required for thread-only parallelism, only process-based parallelism.
        """
        if self.input_queue is None:
            raise ValueError("Please call self._setup before and connect queues.")
        if self.use_savers:
            _ = self.save(context, full_dump=True)  # Cannot delete queue worker self.
            del self.wrapped  # that will be reloaded in the new thread or process.

        process_safe_context: CX = context
        process_safe_context = process_safe_context.thread_safe()
        ParallelObj = Thread
        if self.use_processes:  # TODO: that was to debug pickles.
            # New process requires trimming the references to other processes
            # when we create many processes: https://stackoverflow.com/a/65749012
            # Important check to avoid this cPython issue to cause a deadlock: https://github.com/python/cpython/issues/79423
            process_safe_context = process_safe_context.process_safe()
            ParallelObj = Process

        if self.use_processes and logging_queue is None:
            self.logging_thread = ParallelLoggingConsumerThread()
            logging_queue = self.logging_thread.logging_queue
            self.logging_thread.start()

        self.running_workers = []
        for i in range(self.n_workers):
            p = ParallelObj(
                target=worker_function,
                name=self.name + f"__worker_function{i}",
                args=(self, process_safe_context, self.use_savers, logging_queue)
            )
            p.daemon = True
            self.running_workers.append(p)
            p.start()

        if self.use_savers:
            self.reload_post_saving(context)

    def reload_post_saving(self, context: CX) -> 'ParallelWorkersWrapper':
        self._assert(
            self.use_savers,
            "Attempted to reload post saving whilst not using savers."
            "Don't reload when not using savers.",
            context)
        saved_worker: ParallelWorkersWrapper = context.load(self.get_name())
        self.set_step(saved_worker.get_step())

    def join(self):
        """
        Wait for workers to finish at least their capture of the logging calls.
        """
        _ProducerConsumerMixin.join(self)
        if self.logging_thread is not None:
            self.logging_thread.join(timeout=5.0)
            self.logging_thread = None

    def _teardown(self) -> Optional[RecursiveDict]:
        """
        Stop all processes on teardown.
        """
        self.stop()
        _ProducerConsumerMixin._teardown(self)
        return MetaStep._teardown(self)

    def stop(self):
        """
        Stop all of the workers.

        :return:
        """
        if self.use_processes:
            for w in self.running_workers:
                w.terminate()
        self.join()
        self.running_workers = []
        self.consumers = []


QueuedPipelineStepsTuple = Union[
    BaseTransformer,  # step
    Tuple[int, BaseTransformer],  # (n_workers, step)
    Tuple[str, BaseTransformer],  # (step_name, step)
    Tuple[str, int, BaseTransformer],  # (step_name, n_workers, step)
    Tuple[str, int, int, BaseTransformer],  # (step_name, n_workers, max_queued_minibatches, step)
]
FullQueuedPipelineStepsTuple = Tuple[str, int, int, List[Tuple], BaseTransformer]


class BaseQueuedPipeline(MiniBatchSequentialPipeline):
    """
    Sub class of :class:`Pipeline`.
    Transform data in many pipeline steps at once in parallel in the pipeline using multiprocessing Queues.

    Example usage :

    .. code-block:: python

        # Multiple ways of specifying the steps tuples exists to do various things:
        # step name, step
        p = SequentialQueuedPipeline([
            ('step_a', Identity()),
            ('step_b', Identity()),
        ], n_workers=1, batch_size=10, max_queued_minibatches=10)

        # step name, number of workers, step
        p = SequentialQueuedPipeline([
            ('step_a', 1, Identity()),
            ('step_b', 1, Identity()),
        ], batch_size=10, max_queued_minibatches=10)

        # step name, number of workers, and max size
        p = SequentialQueuedPipeline([
            ('step_a', 1, 10, Identity()),
            ('step_b', 1, 10, Identity()),
        ], batch_size=10)

        # step name, number of workers for each step, and additional argument for each worker
        p = SequentialQueuedPipeline([
            ('step_a', 1, [('host', 'host1'), ('host', 'host2')], 10, Identity())
        ], batch_size=10)

        # step name, number of workers for each step, additional argument for each worker, and max size
        p = SequentialQueuedPipeline([
            ('step_a', 1, [('host', 'host1'), ('host', 'host2')], 10, Identity())
        ], batch_size=10)

        # It's also possible to do parallel feature unions:
        n_workers = 4
        worker_arguments = [('hyperparams', HyperparameterSamples({'multiply_by': 2})) for _ in range(n_workers)]
        p = ParallelQueuedFeatureUnion([
            ('1', n_workers, worker_arguments, MultiplyByN()),
        ], batch_size=10, max_queued_minibatches=5)
        outputs = p.transform(list(range(100)))

    :param steps: pipeline steps.
    :param batch_size: number of elements to combine into a single batch.
    :param n_workers_per_step: number of workers to spawn per step.
    :param max_queued_minibatches: max number of batches inside the processing queue between the workers.
    :param data_joiner: transformer step to join streamed batches together at the end of the pipeline.
    :param use_processes: use processes instead of threads for parallel processing. multiprocessing.Process is used by default.
    :param use_savers: use savers to serialize steps for parallel processing. Recommended if using processes instead of threads.
    :param keep_incomplete_batch: (Optional.) A bool that indicates whether
    or not the last batch should be dropped in the case it has fewer than
    `batch_size` elements; the default behavior is to keep the smaller batch.
    :param default_value_data_inputs: expected_outputs default fill value
    for padding and values outside iteration range, or :class:`~neuraxle.data_container.StripAbsentValues`
    to trim absent values from the batch
    :param default_value_expected_outputs: expected_outputs default fill value
    for padding and values outside iteration range, or :class:`~neuraxle.data_container.StripAbsentValues`
    to trim absent values from the batch
    """

    def __init__(
            self,
            steps: List[QueuedPipelineStepsTuple],
            batch_size: int,
            n_workers_per_step: int = None,
            max_queued_minibatches: int = None,
            data_joiner: BaseTransformer = None,
            use_processes: bool = False,
            use_savers: bool = False,
            keep_incomplete_batch: bool = True,
            default_value_data_inputs: Union[Any, StripAbsentValues] = None,
            default_value_expected_outputs: Union[Any, StripAbsentValues] = None,
    ):
        self.batch_size: int = batch_size
        self.n_workers_per_step: int = n_workers_per_step
        self.max_queued_minibatches: int = max_queued_minibatches or 0
        if data_joiner is None:
            # Note that the data joiner differs from the workers joiner, in the sense that the
            # data joiner is applied after the data is joined by the workers joiner.
            data_joiner = NumpyConcatenateOuterBatch()
        self.data_joiner: BaseTransformer = data_joiner
        self.use_processes: bool = use_processes
        self.use_savers: bool = use_savers
        self.keep_incomplete_batch: bool = keep_incomplete_batch
        self.default_value_data_inputs: Union[Any, StripAbsentValues] = default_value_data_inputs
        self.default_value_expected_outputs: Union[Any, StripAbsentValues] = default_value_expected_outputs

        self.is_pipeline_connected = False

        MiniBatchSequentialPipeline.__init__(
            self,
            steps=self._parallel_wrap_all_steps_tuples(steps),
            batch_size=batch_size,
            keep_incomplete_batch=keep_incomplete_batch,
            default_value_data_inputs=default_value_data_inputs,
            default_value_expected_outputs=default_value_expected_outputs
        )
        self._refresh_steps()

        self.logging_thread: Optional[ParallelLoggingConsumerThread] = None

    def _parallel_wrap_all_steps_tuples(self, all_step_tuples: NamedStepsList) -> NamedStepsList:
        """
        Wrap each step by a :class:`QueueWorker` to  allow data to flow in many pipeline steps at once in parallel.

        :param steps: (name, n_workers, step)
        :type steps: NameNWorkerStepTupleList
        :return: steps as tuple
        """
        steps_as_tuple: NamedStepsList = []
        for step_tuple in all_step_tuples:
            parallel_worker_wrapper = self._parallel_wrap_step_tuple(step_tuple)
            steps_as_tuple.append((parallel_worker_wrapper.name, parallel_worker_wrapper))

        steps_as_tuple.append(("WorkersJoiner", WorkersJoiner(batch_size=self.batch_size)))
        return steps_as_tuple

    def _parallel_wrap_step_tuple(self, step_tuple: QueuedPipelineStepsTuple):
        name, n_workers, max_queued_minibatches, actual_step = self._parse_step_tuple(
            step_tuple)

        return ParallelWorkersWrapper(
            actual_step,
            n_workers=n_workers,
            use_processes=self.use_processes,
            max_queued_minibatches=max_queued_minibatches,
            use_savers=self.use_savers
        ).set_name(name)

    def _parse_step_tuple(self, step_tuple: QueuedPipelineStepsTuple) -> FullQueuedPipelineStepsTuple:
        """
        Return all params necessary to create the QueuedPipeline for the given step.

        :param step_tuple: the un-parsed tuple of steps
        :type step: QueuedPipelineStepsTupleList
        :return: a tuple of (name, n_workers, max_queued_minibatches, actual_step)
        :rtype: Tuple[str, int, int, BaseStep]
        """
        if isinstance(step_tuple, BaseTransformer):
            step_tuple = (step_tuple,)
        elif len(step_tuple) > 5:
            raise Exception(f'Invalid Queued Pipeline Steps Shape: {step_tuple}.')
        actual_step: BaseTransformer = step_tuple[-1]

        # Default values before parse, if missing:
        name: str = actual_step.name
        n_workers: int = max(1, self.n_workers_per_step or 1)
        max_queued_minibatches: int = self.max_queued_minibatches

        if len(step_tuple) == 2:
            if isinstance(step_tuple[0], str):
                name = step_tuple[0]
            else:
                n_workers = step_tuple[0]
        elif len(step_tuple) >= 3:
            name = step_tuple[0]
            n_workers = step_tuple[1]
            if len(step_tuple) == 4:
                max_queued_minibatches = step_tuple[2]

        return (name, n_workers, max_queued_minibatches, actual_step)

    def _will_process(
        self, data_container: DACT, context: CX
    ) -> Tuple[DACT, CX]:
        """
        Setup streaming pipeline before any handler methods.

        :param data_container: data container
        :param context: execution context
        :return:
        """
        self._setup(context=context)
        context = context.synchroneous()
        return data_container.copy(), context

    def _setup(self, context: CX = None) -> 'BaseTransformer':
        """
        Connect the queued workers together so that the data can correctly flow through the pipeline.

        :param context: execution context
        :return: step
        :rtype: BaseStep
        """
        workers_joiner: WorkersJoiner = self.joiner
        workers_joiner._setup(context=context)
        for step in self.body:
            step: ParallelWorkersWrapper = step
            step._setup(context=context)

        self._connect_queued_pipeline()

        return MiniBatchSequentialPipeline._setup(self, context=context)

    def fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple[Pipeline, DACT]:
        """
        Fit transform sequentially if any step is fittable, such as with :class:`MiniBatchSequentialPipeline`. Otherwise transform in parallel as it should.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return:
        """
        has_fittable_step: bool = False
        for step in self.body:
            if isinstance(step.get_step(), _FittableStep) and not isinstance(step.get_step(), NonFittableMixin):
                has_fittable_step = True

        if has_fittable_step:
            # will use MiniBatchSequentialPipeline parent:
            self.is_invalidated = True
            return super().fit_transform_data_container(data_container, context)
        else:
            # Use the parallel transform without fitting:
            data_container = self.transform_data_container(data_container, context)
            data_container = self._did_transform(data_container, context)
            return self, data_container

    def transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        """
        Transform data container

        :param data_container: data container to transform.
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        """
        logging_queue: Optional[Queue] = None
        if self.use_processes:
            self.logging_thread = ParallelLoggingConsumerThread()
            logging_queue = self.logging_thread.logging_queue
            self.logging_thread.start()

        for step in self.values():
            if step.input_queue is None:
                raise ValueError("Please connect queues and do the self._setup before attempting to join.")

        # start steps with parallelized context.
        for step in self.body:
            step: ParallelWorkersWrapper = step
            step.start(context, logging_queue)

        # prepare minibatch iterator:
        data_container.set_ids(data_container.ids)
        minibatch_iterator: Iterable[DACT[IDT, DIT, EOT]] = data_container.minibatches(
            batch_size=self.batch_size,
            keep_incomplete_batch=self.keep_incomplete_batch,
            default_value_data_inputs=self.default_value_data_inputs,
            default_value_expected_outputs=self.default_value_expected_outputs
        )

        # send batches to input queues and label queue output expected summaries.
        n_minibatches_per_worker = 0
        for minibatch_index, minibatch_dact in enumerate(minibatch_iterator):
            task = QueuedMinibatchTask(minibatch_dact, self.name)

            self._dispatch_minibatch_to_consumer_workers(
                minibatch_index=minibatch_index, task=task)

            n_minibatches_per_worker += 1

        # join output queues.
        n_workers = self.get_n_workers_to_join()
        workers_joiner: WorkersJoiner = self.joiner
        workers_joiner.set_join_quantities(n_workers, n_minibatches_per_worker)
        data_container = workers_joiner.join_workers(data_container, context)

        # for step in self.body:
        #     step: ParallelWorkersWrapper = step
        #     step.join()

        return data_container

    def _did_transform(self, data_container: DACT, context: CX) -> DACT:
        """
        Stop all of the workers after transform. Also, join the data using self.data_joiner.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        :rtype: DataContainer
        """
        for step in self.body:
            step: ParallelWorkersWrapper = step
            step.stop()
        if self.logging_thread is not None:
            self.logging_thread.join(timeout=5.0)
            self.logging_thread = None

        return self.data_joiner.handle_transform(data_container, context)

    def _did_process(self, data_container: PredsDACT, context: CX) -> PredsDACT:
        self._disconnect_queued_pipeline()
        return super()._did_process(data_container, context)

    @abstractmethod
    def get_n_workers_to_join(self) -> int:
        """
        Get the total number of terminal steps at the end of each row of queued workers.
        """
        raise NotImplementedError()

    @abstractmethod
    def _connect_queued_pipeline(self):
        """
        Connect all the queued workers together so that the data can flow through each step.

        :return:
        """
        if not self.is_pipeline_connected:
            # register queues and summaries in workers_joiner...
            self.is_pipeline_connected = True
        raise NotImplementedError()

    def _disconnect_queued_pipeline(self):
        if self.is_pipeline_connected:
            workers_joiner: WorkersJoiner = self.joiner
            workers_joiner.join()
            for step in self.body:
                step: ParallelWorkersWrapper = step
                step.join()
        self.is_pipeline_connected = False

    @abstractmethod
    def _dispatch_minibatch_to_consumer_workers(self, minibatch_index: int, task: QueuedMinibatchTask):
        """
        Send batches to queued pipeline. It is blocking if there is no more space available in the multiprocessing queues.
        Workers might return batches in a different order, but the queue joiner will reorder them at the end.
        The queue joiner will use the summary ids to reorder all of the received batches.

        :param batch_index: batch index
        :param data_container: data container batch
        :return:
        """
        raise NotImplementedError()


class SequentialQueuedPipeline(BaseQueuedPipeline):
    """
    Using :class:`QueueWorker`, run all steps sequentially even if they are in separate processes or threads.
    This is a parallel pipeline that uses a queue to communicate between the steps, and which parallelizes the steps
    using many workers in different processes or threads.

    This pipeline is useful when the steps are independent of each other and can be run in parallel. This is especially
    the case when the steps are not fittable, such as inheriting from the :class:`~neuraxle.base.NonFittableMixin`.
    Otherwise, fitting may not be parallelized, although the steps can be run in parallel for the transformation.

    .. seealso::
        :class:`~neuraxle.pipeline.BasePipeline`,
        :func:`~neuraxle.data_container.DataContainer.minibatches`,
        :class:`~neuraxle.data_container.StripAbsentValues`,
        :class:`QueueWorker`,
        :class:`BaseQueuedPipeline`,
        :class:`ParallelQueuedPipeline`,
    """

    def get_n_workers_to_join(self) -> int:
        return 1

    def _connect_queued_pipeline(self):
        """
        Sequentially connect of the queued workers as producers and consumers.

        :return:
        """
        if not self.is_pipeline_connected:
            for i, (name, consumer) in enumerate(self[1:]):
                producer: _ProducerConsumerMixin = self[i]
                producer.register_consumer(consumer)
            self.is_pipeline_connected = True

    def _dispatch_minibatch_to_consumer_workers(self, minibatch_index: int, task: QueuedMinibatchTask):
        """
        Send batches to process to the first queued worker.

        :param batch_index: batch index
        :param data_container: data container batch
        :return:
        """
        workers_joiner: WorkersJoiner = self.joiner

        # TODO: extract method.
        last_consumer_name: str = self.steps_as_tuple[-2][0]
        workers_joiner.append_terminal_summary(last_consumer_name, task)

        first_consumer: _ProducerConsumerMixin = self[0]
        first_consumer.put_minibatch_produced(task)


class ParallelQueuedFeatureUnion(BaseQueuedPipeline):
    """
    Using :class:`QueueWorker`, run all steps in parallel using QueueWorkers.

    .. seealso::
        :class:`QueueWorker`,
        :class:`BaseQueuedPipeline`,
        :class:`SequentialQueuedPipeline`,
    """

    def get_n_workers_to_join(self):
        n_worker_wrappers = len(self) - 1
        return n_worker_wrappers

    def _connect_queued_pipeline(self):
        """
        Connect the queue joiner to all of the queued workers to process data in parallel.

        :return:
        """
        if not self.is_pipeline_connected:
            joiner_consumer: _ProducerConsumerMixin = self.joiner
            for producer in self.body:
                producer: _ProducerConsumerMixin = producer
                producer.register_consumer(joiner_consumer)
            self.is_pipeline_connected = True

    def _dispatch_minibatch_to_consumer_workers(self, minibatch_index: int, task: QueuedMinibatchTask):
        """
        In the case of the feature union, sending a batch to the workers is done by sending the batch
        to each of the workers that will work in parallel to consume the same copy sent to all.
        """
        # TODO: task index in task.
        workers_joiner: WorkersJoiner = self.joiner
        for consumer in self.body:
            consumer: ParallelWorkersWrapper = consumer
            workers_joiner.append_terminal_summary(consumer.get_name(), task)
            consumer.put_minibatch_produced(task)


MiniBatchDataContainer = ListDataContainer[IDT, DIT, EOT]
SummaryIDStr = str  # Picture this as an IDT where each ID is a summary ID
MetaListDataContainer = ListDataContainer[
    List[SummaryIDStr],
    MiniBatchDataContainer,
    List[None]
]


class WorkersJoiner(_ProducerConsumerMixin, Joiner):
    """
    Consume the results of the other :class:`_ProducerConsumerMixin` workers to join their data.
    Also do error handling.
    """

    def __init__(self, batch_size: int, n_worker_wrappers_to_join: int = None):
        # TODO: the present class could be renamed to WorkersTerminalConsumer or so. The [-1] item of the pipeline should be the data joiner instead.
        Joiner.__init__(self, batch_size=batch_size)
        _ProducerConsumerMixin.__init__(self)
        self.n_workers = n_worker_wrappers_to_join
        self.names: List[str] = []
        self.summaries: List[str] = []

    def _setup(self, context: 'CX' = None) -> Optional[RecursiveDict]:
        _ProducerConsumerMixin._setup(self, context)
        return Joiner._setup(self, context)

    def _teardown(self) -> Optional[RecursiveDict]:
        """
        Properly clean queue, summary ids, and results during teardown.

        :return: teardowned self
        """
        self.names = []
        self.summaries = []
        _ProducerConsumerMixin._teardown(self)
        return Joiner._teardown(self)

    def set_join_quantities(self, n_workers: int, n_minibatches_per_worker: int):
        self.n_workers = n_workers
        self.n_workers_remaining = n_workers
        self.n_minibatches_per_worker = n_minibatches_per_worker
        self.remaining_batches_per_workers: Dict[str, int] = defaultdict(lambda: self.n_minibatches_per_worker)

    def append_terminal_summary(self, name: str, task: QueuedMinibatchTask):
        """
        Append the summary id of the worker to the list of summaries.

        :param name: name of the worker
        :param task: task
        :return:
        """
        self.names.append(name)
        self.summaries.append(task.minibatch_dact.get_ids_summary())

    def join_workers(self, original_dact: DACT, sync_context: CX) -> DACT:
        """
        Return the accumulated results of the workers.

        :return: transformed data container
        :rtype: DataContainer
        """
        if self.input_queue is None:
            raise ValueError("Please connect queues and do the self._setup before attempting to join.")

        if self.n_minibatches_per_worker == 0:
            return original_dact

        # Gather minibatch tasks to consume and organize them per step name:
        step_to_minibatches_dacts: Dict[str, MetaListDataContainer] = \
            self._consume_enqueued_minibatches(sync_context)

        # Sort step_to_minibatches_dacts by step_name index in the step names list as like in the loop below but based on the order of steps. In the case of a pipeline, there will be only the last step anyway so the sort will return without even sorting. For feature union, there are many.
        list_dact: List[ListDataContainer] = \
            self._merge_minibatches(step_to_minibatches_dacts)

        _ProducerConsumerMixin.join(self)

        return original_dact.set_data_inputs(list_dact)

    def _consume_enqueued_minibatches(self, sync_context) -> Dict[str, MetaListDataContainer]:

        step_to_minibatches_dacts: Dict[str, MetaListDataContainer] = \
            defaultdict(MiniBatchDataContainer.empty)
        # shape: [worker, minibatches, dact_of_a_minibatch]
        # The shape is confusing because the ListDataContainer contains itself some other ListDataContainer.

        while self.n_workers_remaining > 0:
            # Note: the call here is most likely the reason for
            #       deadlocks if something weird happened in the worker_function.

            try:
                task: QueuedMinibatchTask = self._get_minibatch_to_consume()  # breakpoint here.

                if task.is_error():
                    task: MinibatchError = task
                    err = task.get_err()
                    raise err from err
                else:
                    step_to_minibatches_dacts[task.worker_name].append_data_container_in_data_inputs(
                        task.minibatch_dact)

                    self.remaining_batches_per_workers[task.worker_name] -= 1
                    if self.remaining_batches_per_workers[task.worker_name] == 0:
                        self.n_workers_remaining -= 1
            except queue.Empty:
                continue

        return step_to_minibatches_dacts

    def _merge_minibatches(self, step_to_minibatches_dacts: Dict[str, MetaListDataContainer]) -> List[ListDataContainer]:
        step_to_minibatches_dacts = OrderedDict(sorted(
            step_to_minibatches_dacts.items(),
            key=lambda str_dact_tup: self.names.index(str_dact_tup[0])
        ))

        # Join all the results based on step name:
        list_dact: List[ListDataContainer] = []  # shape: [worker, regular_dact]

        for minibatches_list_dacts in step_to_minibatches_dacts.values():
            minibatches_list_dacts: MetaListDataContainer = minibatches_list_dacts

            # reorder results by ids of summary
            # TODO: check if could use original dact instead. or other batch count int number, that would be more efficient.
            minibatches_list_dacts.data_inputs.sort(key=lambda dc: self.summaries.index(dc.get_ids_summary()))

            sorted_step_dact = ListDataContainer.empty()
            for minibatch_dact in minibatches_list_dacts.data_inputs:
                minibatch_dact: MiniBatchDataContainer = minibatch_dact
                sorted_step_dact.extend(minibatch_dact)

            list_dact.append(sorted_step_dact)

        return list_dact
