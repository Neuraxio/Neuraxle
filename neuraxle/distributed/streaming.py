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
import logging
import time
from abc import abstractmethod
from collections import defaultdict
from multiprocessing import Lock, Process, Queue, RLock
from threading import Thread
import traceback
from typing import Any, Dict, Iterable, List, Tuple, Union

from neuraxle.base import BaseSaver, BaseTransformer
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import (MetaStep, MixinForBaseTransformer, NamedStepsList,
                           NonFittableMixin, _FittableStep)
from neuraxle.data_container import DACT, IDT, EOT, DIT, ListDataContainer, StripAbsentValues
from neuraxle.hyperparams.space import RecursiveDict
from neuraxle.logging.logging import ParallelLoggingConsumerThread
from neuraxle.pipeline import Joiner, MiniBatchSequentialPipeline, Pipeline
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch


class _ProducerConsumerStepSaver(BaseSaver):
    """
    Saver for :class:`_ProducerConsumerMixin`.
    """

    def save_step(self, step: BaseTransformer, context: 'CX') -> BaseTransformer:
        step: _ProducerConsumerMixin = step  # typing.
        step.input_queue = None
        step.consumers = []
        return step

    def can_load(self, step: BaseTransformer, context: 'CX') -> bool:
        return True

    def load_step(self, step: 'BaseTransformer', context: 'CX') -> 'BaseTransformer':
        step: _ProducerConsumerMixin = step  # typing.
        step.input_queue = Queue()
        return step


class _ProducerConsumerMixin(MixinForBaseTransformer):
    """
    A class to represent a step that can receive minibatches from a producer in its queue to consume them.
    Once minibatches are consumed by the present step, they are produced back to the next consumers in line.

    The Queue in self is the one at the entry to be consumed by self. The output queue is external, in the registred consumers.

    Therefore, in this sense, consumers and producers are both themselves instances of _HasQueueMixin and play the two roles unless they are the first an lasts of their multiprocessing pipelines.
    They will thus be used to consume the produced tasks added to them in their other threads or processes.
    """

    def __init__(self, input_queue: Queue = None):
        # TODO: two class for this: a producer class and a consumer class. Also remove all references to the word "observer"
        MixinForBaseTransformer.__init__(self)
        self.input_queue: Queue = input_queue or Queue()
        self.consumers: List[Queue] = []
        self.savers.append(_ProducerConsumerStepSaver())

    def _teardown(self):
        self.input_queue = None
        # TODO: queue never setupped again after? Why?
        return self

    def register_consumer(self, recepient: '_ProducerConsumerMixin') -> '_ProducerConsumerMixin':
        """
        Add a consumer to self.consumers so that when self produces a minibatch, it allws consumers to consume it.
        """
        self.consumers.append(recepient.input_queue)
        return self

    def put_minibatch_produced_to_next_consumers(self, value: 'QueuedMinibatch'):
        """
        Push a minibatch to all subsequent consumers to allow them to consume it.
        """
        for consumers_of_self in self.consumers:
            consumers_of_self: _ProducerConsumerMixin  # typing...

            consumers_of_self.put(value)

    def put_minibatch_produced(self, value: 'QueuedMinibatch'):
        """
        Put a minibatch in queue.
        The caller of this method is the producer of the minibatch and is external to self.
        """
        self.input_queue.put(value)  # TODO: possibly copy DACT here.

    def _get_minibatch_to_consume(self) -> 'QueuedMinibatch':
        """
        Get last minibatch in queue. The caller of this method is probably self,
        that is why the method is private (starts with an underscore).
        """
        return self.input_queue.get()


class QueuedMinibatch:
    """
    Data object to contain the minibatch processed by producers and consumers.
    """

    def __init__(self, minibatch_dact: DACT, step_name: str = None):
        self.step_name: str = step_name
        self.minibatch_dact: ListDataContainer = minibatch_dact

    def is_terminal(self) -> bool:
        return False

    def is_error(self) -> bool:
        return False

    def terminal(self) -> 'LastQueuedMinibatch':
        return LastQueuedMinibatch(
            minibatch_dact=self.minibatch_dact,
            step_name=self.step_name)

    def error(self, error: Exception, stack_trace=None) -> 'LastQueuedMinibatchWithError':
        return LastQueuedMinibatchWithError(
            minibatch_dact=self.minibatch_dact,
            step_name=self.step_name,
            error=error,
            stack_trace=stack_trace)


class LastQueuedMinibatch(QueuedMinibatch):
    """
    This is a :class:`QueuedMinibatch` that is the last one sent in the queue.
    """

    def is_terminal(self) -> bool:
        return True


class LastQueuedMinibatchWithError(LastQueuedMinibatch):
    """
    Data object to represent an error in a minibatch.
    """

    def __init__(self, minibatch_dact: DACT, step_name: str, error: Exception, stack_trace=None):
        super().__init__(minibatch_dact, step_name)
        self.error: Exception = error
        self.stack_trace = stack_trace

    def is_error(self) -> bool:
        return True


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
        additional_worker_arguments: List = None,
        use_savers: bool = False
    ):
        if not additional_worker_arguments:
            additional_worker_arguments = [[] for _ in range(n_workers)]

        MetaStep.__init__(self, wrapped)
        max_queued_minibatches = max_queued_minibatches or 0
        _ProducerConsumerMixin.__init__(self, Queue(maxsize=max_queued_minibatches))
        self.n_workers: int = n_workers
        self.use_processes: bool = use_processes
        self.additional_worker_arguments = additional_worker_arguments
        self.use_savers = use_savers

        self.running_workers: List[Process] = []
        self.logging_thread: ParallelLoggingConsumerThread = None

    def __getstate__(self):
        """
        This class, upon being forked() to a new process with pickles,
        should not copy references to other threads or processes.
        """
        state = self.__dict__.copy()
        state['running_workers'] = []  # delete self.workers in the process fork's pickling.
        state["logging_thread"] = None
        return state

    def start(self, context: CX):
        """
        Start multiple processes or threads with the worker function as a target.
        These workers will consume minibatches from the queue and produce them on the next queue(s).
        They are started as multiprocessing daemons, so that they will not block the main
        process if there is an error requiring to exit.
        """
        if self.use_savers:
            _ = self.save(context, full_dump=True)  # Cannot delete queue worker self.
            del self.wrapped  # that will be reloaded in the new thread or process.

        process_safe_context = context
        thread_safe_lock: RLock = context.synchroneous()
        ParallelObj = Thread
        logging_queue: Queue = None
        if self.use_processes:
            # New process requires trimming the references to other processes
            # when we create many processes: https://stackoverflow.com/a/65749012
            thread_safe_lock, logging_thread, process_safe_context = context.process_safe()
            logging_queue: Queue = logging_thread.logging_queue
            ParallelObj = Process

        self.running_workers = []
        for _, worker_arguments in zip(range(self.n_workers), self.additional_worker_arguments):
            p = ParallelObj(
                target=worker_function,
                args=(self, thread_safe_lock, process_safe_context, self.use_savers, logging_queue, worker_arguments)
            )
            p.daemon = True
            self.running_workers.append(p)
            p.start()

        if self.use_processes:
            self.logging_thread = logging_thread
            self.logging_thread.start()

    def join(self):
        """
        Wait for workers to finish at least their capture of the logging calls.
        """
        # for worker in self.running_workers:
        #     worker.join()
        if self.logging_thread is not None:
            self.logging_thread.join(timeout=5.0)

    def _teardown(self):
        """
        Stop all processes on teardown.

        :return: teardowned self
        """
        self.stop()
        return self

    def stop(self):
        """
        Stop all of the workers.

        :return:
        """
        if self.use_processes:
            [w.terminate() for w in self.running_workers]
            self.logging_thread.join(timeout=5.0)
        self.running_workers = []
        self.logging_thread = None
        self.consumers = []


def worker_function(
    parallel_worker_wrapper: ParallelWorkersWrapper,
    shared_lock: Lock,
    context: CX,
    use_savers: bool,
    logging_queue: Queue,
    additional_worker_arguments
):
    """
    Worker function that transforms the items inside the queue of items to process.

    :param queue_worker: step to transform
    :param context: execution context
    :param use_savers: use savers
    :param additional_worker_arguments: any additional arguments that need to be passed to the workers
    :return:
    """
    name = parallel_worker_wrapper.name
    try:
        context.restore_lock(shared_lock)
        if use_savers:
            saved_queue_worker: ParallelWorkersWrapper = context.load(parallel_worker_wrapper.get_name())
            parallel_worker_wrapper.set_step(saved_queue_worker.get_step())
            # TODO: what happens to the list of next consumers of the _ProducerConsumerMixin in self?
        step = parallel_worker_wrapper.get_step()

        additional_worker_arguments = tuple(
            additional_worker_arguments[i: i + 2] for i in range(0, len(additional_worker_arguments), 2)
        )

        for argument_name, argument_value in additional_worker_arguments:
            step.__dict__.update({argument_name: argument_value})

        # TODO: extract this next logging_queue if to a separate function in logging.py:
        if logging_queue is not None:
            queue_handler = logging.handlers.QueueHandler(logging_queue)
            root = logging.getLogger()
            root.setLevel(logging.DEBUG)
            root.addHandler(queue_handler)

        while True:
            try:
                task: QueuedMinibatch = parallel_worker_wrapper._get_minibatch_to_consume()

                if task.is_error():
                    parallel_worker_wrapper.put_minibatch_produced_to_next_consumers(task)
                    break

                data_container = step.handle_transform(task.minibatch_dact, context)
                task.step_name = name
                task.minibatch_dact = data_container
                parallel_worker_wrapper.put_minibatch_produced_to_next_consumers(task)

                if task.is_terminal():
                    break

            except Exception as err:
                context.flow.log_error(err)
                stack_trace = traceback.format_exc()
                task = task.error(err, stack_trace)
                parallel_worker_wrapper.put_minibatch_produced_to_next_consumers(task)
            finally:
                time.sleep(0.005)  # Sleeping here empirically seems to improve overall computation time on MacOS M1.
    except Exception as err:
        context.flow.log_error(err)
        stack_trace = traceback.format_exc()
        parallel_worker_wrapper.put_minibatch_produced_to_next_consumers(
            LastQueuedMinibatchWithError(
                None, name, err, stack_trace
            )
        )


QueuedPipelineStepsTuple = Union[
    BaseTransformer,  # step
    Tuple[int, BaseTransformer],  # (n_workers, step)
    Tuple[str, BaseTransformer],  # (step_name, step)
    Tuple[str, int, BaseTransformer],  # (step_name, n_workers, step)
    Tuple[str, int, int, BaseTransformer],  # (step_name, n_workers, max_queued_minibatches, step)
    Tuple[str, int, List[Tuple], BaseTransformer],  # (step_name, n_workers, additional_worker_arguments, step)
    Tuple[str, int, List[Tuple], BaseTransformer]  # (step_name, n_workers, additional_worker_arguments, step)
]


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
    :param use_processes: use processes instead of threads for parallel processing. multiprocessing.context.Process is used by default.
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
        self.max_queued_minibatches: int = max_queued_minibatches
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

        MiniBatchSequentialPipeline.__init__(
            self,
            steps=self._parallel_wrap_all_steps_tuples(steps),
            batch_size=batch_size,
            keep_incomplete_batch=keep_incomplete_batch,
            default_value_data_inputs=default_value_data_inputs,
            default_value_expected_outputs=default_value_expected_outputs
        )
        self._refresh_steps()

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
        name, n_workers, additional_worker_arguments, max_queued_minibatches, actual_step = self._parse_step_tuple(
            step_tuple)

        return ParallelWorkersWrapper(
            actual_step,
            n_workers=n_workers,
            use_processes=self.use_processes,
            max_queued_minibatches=max_queued_minibatches,
            additional_worker_arguments=additional_worker_arguments,
            use_savers=self.use_savers
        ).set_name('QueueWorker{}'.format(name))

    def _parse_step_tuple(self, step_tuple: QueuedPipelineStepsTuple) -> Tuple[str, int, int, BaseTransformer]:
        """
        Return all params necessary to create the QueuedPipeline for the given step.

        :param step_tuple: the un-parsed tuple of steps
        :type step: QueuedPipelineStepsTupleList
        :return: a tuple of (name, n_workers, max_queued_minibatches, actual_step)
        :rtype: Tuple[str, int, int, BaseStep]
        """
        if isinstance(step_tuple, BaseTransformer):
            actual_step = step_tuple
            name = step_tuple.name
            max_queued_minibatches = self.max_queued_minibatches
            n_workers = self.n_workers_per_step
            additional_arguments = []
        elif len(step_tuple) == 2:
            if isinstance(step_tuple[0], str):
                name, actual_step = step_tuple
                n_workers = self.n_workers_per_step
            else:
                n_workers, actual_step = step_tuple
                name = actual_step.name
            max_queued_minibatches = self.max_queued_minibatches
            additional_arguments = []
        elif len(step_tuple) == 3:
            name, n_workers, actual_step = step_tuple
            max_queued_minibatches = self.max_queued_minibatches
            additional_arguments = []
        elif len(step_tuple) == 4:
            if isinstance(step_tuple[2], Iterable):
                name, n_workers, additional_arguments, actual_step = step_tuple
                max_queued_minibatches = self.max_queued_minibatches
            else:
                name, n_workers, max_queued_minibatches, actual_step = step_tuple
                additional_arguments = []
        elif len(step_tuple) == 5:
            name, n_workers, additional_arguments, max_queued_minibatches, actual_step = step_tuple
        else:
            raise Exception('Invalid Queued Pipeline Steps Shape.')

        return name, n_workers, additional_arguments, max_queued_minibatches, actual_step

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
        return data_container.copy(), context  # TODO: copy here, really?

    def _setup(self, context: CX = None) -> 'BaseTransformer':
        """
        Connect the queued workers together so that the data can correctly flow through the pipeline.

        :param context: execution context
        :return: step
        :rtype: BaseStep
        """
        if not self.is_initialized:
            self._connect_queued_pipeline()
        super()._setup(context=context)
        return RecursiveDict()

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
        for _, step in self[:-1]:
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
        workers_joiner: WorkersJoiner = self[-1]
        n_branches = self.get_n_parallel_workers_branches(data_container)
        workers_joiner.set_n_parallel_workers_branches(n_branches)

        # start steps with parallelized context.
        context.synchroneous()
        for step in list(self.values())[:-1]:
            step: ParallelWorkersWrapper = step
            step.start(context)

        # prepare minibatch iterator:
        minibatch_iterator: Iterable[DACT[IDT, DIT, EOT]] = data_container.minibatches(
            batch_size=self.batch_size,
            keep_incomplete_batch=self.keep_incomplete_batch,
            default_value_data_inputs=self.default_value_data_inputs,
            default_value_expected_outputs=self.default_value_expected_outputs
        )

        # send batches to input queues and label queue output expected summaries.
        prev_minibatch_index = None
        prev_iter_task: QueuedMinibatch = None
        for minibatch_index, minibatch_dact in enumerate(minibatch_iterator):
            task = QueuedMinibatch(minibatch_dact, self.name)

            if prev_iter_task is not None:
                self._dispatch_minibatch_to_consumer_workers(
                    minibatch_index=prev_minibatch_index, task=prev_iter_task)

            prev_minibatch_index = minibatch_index
            prev_iter_task = task
        prev_iter_task = prev_iter_task.terminal()
        self._dispatch_minibatch_to_consumer_workers(
            minibatch_index=prev_minibatch_index, task=prev_iter_task)

        # join output queues.
        data_container = workers_joiner.join(data_container, context)

        for step in list(self.values())[:-1]:
            # TODO: accessor for these steps?
            step: ParallelWorkersWrapper = step
            step.join()

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
        for name, step in self[:-1]:
            step.stop()

        return self.data_joiner.handle_transform(data_container, context)

    @abstractmethod
    def get_n_parallel_workers_branches(self, data_container) -> int:
        """
        Get the total number of batches that the queue joiner is supposed to receive.

        :param data_container: data container to transform
        :type data_container: DataContainer
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def _connect_queued_pipeline(self):
        """
        Connect all the queued workers together so that the data can flow through each step.

        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def _dispatch_minibatch_to_consumer_workers(self, minibatch_index: int, task: QueuedMinibatch):
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

    def get_n_parallel_workers_branches(self, data_container) -> int:
        """
        Get the number of batches to process.

        :param data_container: data container to transform
        :return: number of batches
        """
        # return data_container.get_n_batches(
        #     batch_size=self.batch_size,
        #     keep_incomplete_batch=self.keep_incomplete_batch
        # )
        return 1

    def _connect_queued_pipeline(self):
        """
        Sequentially connect of the queued workers as producers and consumers.

        :return:
        """
        for i, (name, consumer) in enumerate(self[1:]):
            producer: _ProducerConsumerMixin = self[i]
            producer.register_consumer(consumer)

    def _dispatch_minibatch_to_consumer_workers(self, minibatch_index: int, task: QueuedMinibatch):
        """
        Send batches to process to the first queued worker.

        :param batch_index: batch index
        :param data_container: data container batch
        :return:
        """
        workers_joiner: WorkersJoiner = self[-1]

        # TODO: extract method.
        workers_joiner.summaries.append(task.minibatch_dact.get_ids_summary())

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

    def get_n_parallel_workers_branches(self, data_container):
        """
        Get the number of batches to process by the queue joiner.
        """
        n_parallel_steps = len(self) - 1
        # return data_container.get_n_batches(self.batch_size) * n_parallel_steps
        return n_parallel_steps

    def _connect_queued_pipeline(self):
        """
        Connect the queue joiner to all of the queued workers to process data in parallel.

        :return:
        """
        joiner_consumer: _ProducerConsumerMixin = self[-1]
        for name, producer in self[:-1]:
            producer: _ProducerConsumerMixin = producer
            producer.register_consumer(joiner_consumer)

    def _dispatch_minibatch_to_consumer_workers(self, minibatch_index: int, task: QueuedMinibatch):
        """
        In the case of the feature union, sending a batch to the workers is done by sending the batch
        to each of the workers that will work in parallel to consume the same copy sent to all.
        """
        workers_joiner: WorkersJoiner = self[-1]
        for name, consumer in self[:-1]:
            workers_joiner.summaries.append(task.minibatch_dact.get_ids_summary())
            consumer: _ProducerConsumerMixin = consumer
            consumer.put_minibatch_produced(task)


class WorkersJoiner(_ProducerConsumerMixin, Joiner):
    """
    Observe the results of the :class:`QueueWorker` to append them.
    Synchronize all of the workers together.

    .. seealso::
        :class:`QueuedPipeline`,
        :class:`Observer`,
        :class:`ListDataContainer`,
        :class:`DataContainer`
    """

    def __init__(self, batch_size: int, n_batches: int = None):
        Joiner.__init__(self, batch_size=batch_size)
        _ProducerConsumerMixin.__init__(self)
        self.n_parallel_workers_branches = n_batches
        self.summaries: List[str] = []

    def _teardown(self) -> 'BaseTransformer':
        """
        Properly clean queue, summary ids, and results during teardown.

        :return: teardowned self
        """
        _ProducerConsumerMixin._teardown(self)
        Joiner._teardown(self)
        self.summaries: List[str] = []
        return self

    def set_n_parallel_workers_branches(self, n_batches):
        self.n_parallel_workers_branches = n_batches

    def join(self, original_dact: DACT, sync_context: CX) -> DACT:
        """
        Return the accumulated results received by the on next method of this observer.

        :return: transformed data container
        :rtype: DataContainer
        """
        # Fetch tasks to consume and organize them per step name:
        step_to_minibatches_dacts: Dict[str, ListDataContainer] = defaultdict(ListDataContainer.empty)
        while self.n_parallel_workers_branches > 0:

            task: QueuedMinibatch = self._get_minibatch_to_consume()
            if task.is_terminal():
                task: LastQueuedMinibatch = task
                self.n_parallel_workers_branches -= 1

            if task.is_error():
                task: LastQueuedMinibatchWithError = task
                sync_context.flow.log_error(task.error)
                sync_context.flow.log(task.stack_trace)  # TODO: better stack trace log.
                raise task.error

            step_to_minibatches_dacts[task.step_name].append_data_container_in_data_inputs(task.minibatch_dact)
            # break  # TODO: REVISE THIS.

        # Join all the results based on step name:
        list_dact: List[ListDataContainer] = []  # TODO: revise this type.
        for step_name, minibatches_list_dacts in step_to_minibatches_dacts.items():

            for _dact in minibatches_list_dacts.data_inputs:
                if not isinstance(_dact, DACT):
                    # an exception has been throwned by the worker so reraise it here!
                    exception = _dact
                    sync_context.flow.log_error(exception)
                    raise exception

            # reorder results by ids of summary
            minibatches_list_dacts.data_inputs.sort(key=lambda dc: self.summaries.index(dc.get_ids_summary()))
            sorted_step_dact = ListDataContainer.empty()
            for minibatch_dact in minibatches_list_dacts.data_inputs:
                sorted_step_dact.extend(minibatch_dact)  # TODO: revise these types.

            list_dact.append(sorted_step_dact)

        return original_dact.set_data_inputs(list_dact)


# TODO: have a null event to be passed from the input of the consumers, such as in _dispatch_minibatch_to_consumer_workers, to close the loop upon receiving an error or this null event, instead of relying on the joiner to close the loop with n_steps_left_to_do = 0. Perhaps?
