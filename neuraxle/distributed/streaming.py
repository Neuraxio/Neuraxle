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
import time
import warnings
from abc import abstractmethod
from multiprocessing import Lock, Process, Queue, RLock
from threading import Thread
from typing import Any, Dict, Iterable, List, Tuple, Union

from neuraxle.base import BaseSaver, BaseTransformer
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import (MetaStep, MixinForBaseTransformer, NamedStepsList,
                           NonFittableMixin, _FittableStep)
from neuraxle.data_container import (DACT, AbsentValuesNullObject,
                                     ListDataContainer)
from neuraxle.hyperparams.space import RecursiveDict
from neuraxle.pipeline import Joiner, MiniBatchSequentialPipeline, Pipeline
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch


class ObservableQueueMixin(MixinForBaseTransformer):
    """
    A class to represent a step that can put items in a queue so that the
    step can be used to consume its tasks on its own.

    Once tasks are solved, their results are added to the
    subscribers that were subscribed with the notify call.

    A subscriber is itself an ObservableQueueMixin and will thus be
    used to consume the tasks added to it in probably another thread.

    .. seealso::
        :class:`BaseStep`,
        :class:`QueuedPipelineTask`,
        :class:`QueueWorker`,
        :class:`BaseQueuedPipeline`,
        :class:`ParallelQueuedPipeline`,
        :class:`SequentialQueuedPipeline`
    """

    def __init__(self, queue: Queue):
        MixinForBaseTransformer.__init__(self)
        self.queue = queue
        self.observers = []
        self._add_observable_queue_step_saver()

    def _teardown(self):
        self.queue = None
        return self

    def _add_observable_queue_step_saver(self):
        if not hasattr(self, 'savers'):
            warnings.warn(
                'Please initialize Mixins in the good order. ObservableQueueMixin should be initialized after '
                'Appending the ObservableQueueStepSaver to the savers. Saving might fail.'
            )
            self.savers = [ObservableQueueStepSaver()]
        else:
            self.savers.append(ObservableQueueStepSaver())

    def subscribe_step(self, observer_queue_worker: 'ObservableQueueMixin') -> 'ObservableQueueMixin':
        """
        Subscribe a queue worker to self such that we can notify them to post tasks to put on them.
        The subscribed queue workers get notified when :func:`~neuraxle.distributed.streaming.ObservableQueueMixin.notify_step` is called.
        """
        self.observers.append(observer_queue_worker.queue)
        return self

    def get_task(self) -> 'QueuedPipelineTask':
        """
        Get last item in queue.
        """
        return self.queue.get()

    def put_task(self, value: DACT):
        """
        Put a queued pipeline task in queue.
        """
        self.queue.put(QueuedPipelineTask(step_name=self.name, data_container=value.copy()))

    def notify_step(self, value: DACT):
        """
        Notify all subscribed queue workers to put them some tasks on their queue.
        """
        for observer_queue in self.observers:
            observer_queue.put(QueuedPipelineTask(step_name=self.name, data_container=value))


class QueuedPipelineTask(object):
    """
    Data object to contain the tasks processed by the queued pipeline.
    Attributes: step_name, data_container

    .. seealso::
        :class:`QueueWorker`,
        :class:`BaseQueuedPipeline`,
        :class:`ParallelQueuedPipeline`,
        :class:`SequentialQueuedPipeline`
    """

    def __init__(self, data_container, step_name=None):
        self.step_name = step_name
        self.data_container = data_container


class ObservableQueueStepSaver(BaseSaver):
    """
    Saver for observable queue steps.

    .. seealso::
        :class:`QueueWorker`,
        :class:`neuraxle.base.BaseSaver`,
        :class:`BaseQueuedPipeline`,
        :class:`ParallelQueuedPipeline`,
        :class:`SequentialQueuedPipeline`
    """

    def save_step(self, step: BaseTransformer, context: 'CX') -> BaseTransformer:
        step.queue = None
        step.observers = []
        return step

    def can_load(self, step: BaseTransformer, context: 'CX') -> bool:
        return True

    def load_step(self, step: 'BaseTransformer', context: 'CX') -> 'BaseTransformer':
        step.queue = Queue()
        return step


class QueueWorker(ObservableQueueMixin, MetaStep):
    """
    Start multiple Process or Thread that process items from the queue of batches to process.
    It is both an observable, and observer.
    It notifies the results of the wrapped step handle transform method.
    It receives the next data container to process.

    .. seealso::
        :class:`Observer`,
        :class:`Observable`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`
    """

    def __init__(
            self,
            wrapped: BaseTransformer,
            max_queue_size: int,
            n_workers: int,
            use_processes: bool = True,
            additional_worker_arguments: List = None,
            use_savers: bool = False
    ):
        if not additional_worker_arguments:
            additional_worker_arguments = [[] for _ in range(n_workers)]

        MetaStep.__init__(self, wrapped)
        ObservableQueueMixin.__init__(self, Queue(maxsize=max_queue_size))  # max_queue_size is in batches

        self.use_processes: bool = use_processes
        self.workers: List[Process] = []
        self.n_workers: int = n_workers
        self.observers: List[Queue] = []
        self.additional_worker_arguments = additional_worker_arguments
        self.use_savers = use_savers

    def __getstate__(self):
        """
        This class, upon being forked() to a new process with pickles,
        should not copy references to other threads.
        """
        state = self.__dict__.copy()
        state['workers'] = None
        return state

    def start(self, context: CX):
        """
        Start multiple processes or threads with the worker function as a target.

        :param context: execution context
        :type context: ExecutionContext
        :return:
        """
        if self.use_savers:
            _ = self.save(context, full_dump=True)  # Cannot delete queue worker self.
            del self.wrapped  # queue is deleted

        thread_safe_context = context
        thread_safe_lock: RLock = context.synchroneous()
        parallel_call = Thread
        if self.use_processes:
            # New process requires trimming the references to other processes
            # when we create many processes: https://stackoverflow.com/a/65749012
            thread_safe_lock, thread_safe_context = context.thread_safe()
            parallel_call = Process

        self.workers = []
        for _, worker_arguments in zip(range(self.n_workers), self.additional_worker_arguments):
            p = parallel_call(
                target=worker_function,
                args=(self, thread_safe_lock, thread_safe_context, self.use_savers, worker_arguments)
            )
            p.daemon = True
            p.start()
            self.workers.append(p)

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
            [w.terminate() for w in self.workers]

        self.workers = []
        self.observers = []


def worker_function(queue_worker: QueueWorker, shared_lock: Lock, context: CX, use_savers: bool,
                    additional_worker_arguments):
    """
    Worker function that transforms the items inside the queue of items to process.

    :param queue_worker: step to transform
    :param context: execution context
    :param use_savers: use savers
    :param additional_worker_arguments: any additional arguments that need to be passed to the workers
    :return:
    """

    try:
        context.restore_lock(shared_lock)
        if use_savers:
            saved_queue_worker: QueueWorker = context.load(queue_worker.get_name())
            queue_worker.set_step(saved_queue_worker.get_step())
        step = queue_worker.get_step()

        additional_worker_arguments = tuple(
            additional_worker_arguments[i: i + 2] for i in range(0, len(additional_worker_arguments), 2)
        )

        for argument_name, argument_value in additional_worker_arguments:
            step.__dict__.update({argument_name: argument_value})

        while True:
            try:
                task: QueuedPipelineTask = queue_worker.get_task()
                data_container = step.handle_transform(task.data_container, context)
                queue_worker.notify_step(data_container)
            except Exception as err:
                queue_worker.notify_step(err)
            finally:
                time.sleep(0.005)
    except Exception as err:
        queue_worker.notify_step(err)


QueuedPipelineStepsTuple = Union[
    BaseTransformer,  # step
    Tuple[int, BaseTransformer],  # (n_workers, step)
    Tuple[str, BaseTransformer],  # (step_name, step)
    Tuple[str, int, BaseTransformer],  # (step_name, n_workers, step)
    Tuple[str, int, int, BaseTransformer],  # (step_name, n_workers, max_queue_size, step)
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
        ], n_workers=1, batch_size=10, max_queue_size=10)

        # step name, number of workers, step
        p = SequentialQueuedPipeline([
            ('step_a', 1, Identity()),
            ('step_b', 1, Identity()),
        ], batch_size=10, max_queue_size=10)

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
        ], batch_size=10, max_queue_size=5)
        outputs = p.transform(list(range(100)))

    :param steps: pipeline steps.
    :param batch_size: number of elements to combine into a single batch.
    :param n_workers_per_step: number of workers to spawn per step.
    :param max_queue_size: max number of batches inside the processing queue between the workers.
    :param data_joiner: transformer step to join streamed batches together at the end of the pipeline.
    :param use_processes: use processes instead of threads for parallel processing. multiprocessing.context.Process is used by default.
    :param use_savers: use savers to serialize steps for parallel processing. Recommended if using processes instead of threads.
    :param keep_incomplete_batch: (Optional.) A bool that indicates whether
    or not the last batch should be dropped in the case it has fewer than
    `batch_size` elements; the default behavior is to keep the smaller batch.
    :param default_value_data_inputs: expected_outputs default fill value
    for padding and values outside iteration range, or :class:`~neuraxle.data_container.DataContainer.AbsentValuesNullObject`
    to trim absent values from the batch
    :param default_value_expected_outputs: expected_outputs default fill value
    for padding and values outside iteration range, or :class:`~neuraxle.data_container.DataContainer.AbsentValuesNullObject`
    to trim absent values from the batch

    .. seealso::
        :class:`QueueWorker`,
        :class:`QueueJoiner`,
        :class:`CustomPipelineMixin`,
        :class:`Pipeline`
    """

    def __init__(
            self,
            steps: List[QueuedPipelineStepsTuple],
            batch_size: int,
            n_workers_per_step: int = None,
            max_queue_size: int = None,
            data_joiner: BaseTransformer = None,
            use_processes: bool = False,
            use_savers: bool = False,
            keep_incomplete_batch: bool = True,
            default_value_data_inputs: Union[Any, AbsentValuesNullObject] = None,
            default_value_expected_outputs: Union[Any, AbsentValuesNullObject] = None,
    ):
        if data_joiner is None:
            data_joiner = NumpyConcatenateOuterBatch()
        self.data_joiner = data_joiner
        self.max_queue_size = max_queue_size
        self.n_workers_per_step = n_workers_per_step
        self.use_processes = use_processes
        self.use_savers = use_savers

        self.batch_size: int = batch_size
        self.keep_incomplete_batch: bool = keep_incomplete_batch
        self.default_value_data_inputs: Union[Any, AbsentValuesNullObject] = default_value_data_inputs
        self.default_value_expected_outputs: Union[Any, AbsentValuesNullObject] = default_value_expected_outputs

        MiniBatchSequentialPipeline.__init__(
            self,
            steps=self._initialize_steps_as_tuple(steps),
            batch_size=batch_size,
            keep_incomplete_batch=keep_incomplete_batch,
            default_value_data_inputs=default_value_data_inputs,
            default_value_expected_outputs=default_value_expected_outputs
        )
        self._refresh_steps()

    def _initialize_steps_as_tuple(self, steps: NamedStepsList) -> NamedStepsList:
        """
        Wrap each step by a :class:`QueueWorker` to  allow data to flow in many pipeline steps at once in parallel.

        :param steps: (name, n_workers, step)
        :type steps: NameNWorkerStepTupleList
        :return: steps as tuple
        """
        steps_as_tuple: NamedStepsList = []
        for step in steps:
            queue_worker = self._create_queue_worker(step)
            steps_as_tuple.append((queue_worker.name, queue_worker))

        steps_as_tuple.append(('queue_joiner', QueueJoiner(batch_size=self.batch_size)))

        return steps_as_tuple

    def _create_queue_worker(self, step: QueuedPipelineStepsTuple):
        name, n_workers, additional_worker_arguments, max_queue_size, actual_step = self._get_step_params(step)

        return QueueWorker(
            actual_step,
            n_workers=n_workers,
            use_processes=self.use_processes,
            max_queue_size=max_queue_size,
            additional_worker_arguments=additional_worker_arguments,
            use_savers=self.use_savers
        ).set_name('QueueWorker{}'.format(name))

    def _get_step_params(self, step):
        """
        Return all params necessary to create the QueuedPipeline for the given step.

        :param step: tuple
        :type step: QueuedPipelineStepsTupleList

        :return: return name, n_workers, max_queue_size, actual_step
        :rtype: tuple(str, int, int, BaseStep)
        """
        if isinstance(step, BaseTransformer):
            actual_step = step
            name = step.name
            max_queue_size = self.max_queue_size
            n_workers = self.n_workers_per_step
            additional_arguments = []
        elif len(step) == 2:
            if isinstance(step[0], str):
                name, actual_step = step
                n_workers = self.n_workers_per_step
            else:
                n_workers, actual_step = step
                name = actual_step.name
            max_queue_size = self.max_queue_size
            additional_arguments = []
        elif len(step) == 3:
            name, n_workers, actual_step = step
            max_queue_size = self.max_queue_size
            additional_arguments = []
        elif len(step) == 4:
            if isinstance(step[2], Iterable):
                name, n_workers, additional_arguments, actual_step = step
                max_queue_size = self.max_queue_size
            else:
                name, n_workers, max_queue_size, actual_step = step
                additional_arguments = []
        elif len(step) == 5:
            name, n_workers, additional_arguments, max_queue_size, actual_step = step
        else:
            raise Exception('Invalid Queued Pipeline Steps Shape.')

        return name, n_workers, additional_arguments, max_queue_size, actual_step

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
        return data_container.copy(), context

    def _setup(self, context: CX = None) -> 'BaseTransformer':
        """
        Connect the queued workers together so that the data can correctly flow through the pipeline.

        :param context: execution context
        :return: step
        :rtype: BaseStep
        """
        if not self.is_initialized:
            self.connect_queued_pipeline()
        super()._setup(context=context)
        return RecursiveDict()

    def fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple[Pipeline, DACT]:
        """
        Fit transform sequentially if any step is fittable. Otherwise transform in parallel.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return:
        """
        all_steps_are_not_fittable = True
        for _, step in self[:-1]:
            if isinstance(step.get_step(), _FittableStep) and not isinstance(step.get_step(), NonFittableMixin):
                all_steps_are_not_fittable = False

        if all_steps_are_not_fittable:
            data_container = self.transform_data_container(data_container, context)
            data_container = self._did_transform(data_container, context)
            return self, data_container

        self.is_invalidated = True

        return super().fit_transform_data_container(data_container, context)

    def transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        """
        Transform data container

        :param data_container: data container to transform.
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        """
        joiner = self[-1]
        n_batches = self.get_n_batches(data_container)
        joiner.set_n_batches(n_batches)

        # start steps.
        context.synchroneous()
        for step in list(self.values())[:-1]:
            step.start(context)

        # send batches to input queues and label queue output expected summaries.
        for batch_i, data_container_batch in enumerate(data_container.minibatches(
            batch_size=self.batch_size,
            keep_incomplete_batch=self.keep_incomplete_batch,
            default_value_data_inputs=self.default_value_data_inputs,
            default_value_expected_outputs=self.default_value_expected_outputs
        )):
            self.send_batch_to_queued_pipeline(batch_index=batch_i, data_container=data_container_batch)

        # join output queues.
        data_container = joiner.join(original_data_container=data_container)
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
    def get_n_batches(self, data_container) -> int:
        """
        Get the total number of batches that the queue joiner is supposed to receive.

        :param data_container: data container to transform
        :type data_container: DataContainer
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def connect_queued_pipeline(self):
        """
        Connect all the queued workers together so that the data can flow through each step.

        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def send_batch_to_queued_pipeline(self, batch_index: int, data_container: DACT):
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
        :class:`~neuraxle.data_container.DataContainer.AbsentValuesNullObject`,
        :class:`QueueWorker`,
        :class:`BaseQueuedPipeline`,
        :class:`ParallelQueuedPipeline`,
        :class:`QueueJoiner`,
        :class:`Observer`,
        :class:`Observable`
    """

    def get_n_batches(self, data_container) -> int:
        """
        Get the number of batches to process.

        :param data_container: data container to transform
        :return: number of batches
        """
        return data_container.get_n_batches(
            batch_size=self.batch_size,
            keep_incomplete_batch=self.keep_incomplete_batch
        )

    def connect_queued_pipeline(self):
        """
        Sequentially connect of the queued workers.

        :return:
        """
        for i, (name, step) in enumerate(self[1:]):
            self[i].subscribe_step(step)

    def send_batch_to_queued_pipeline(self, batch_index: int, data_container: DACT):
        """
        Send batches to process to the first queued worker.

        :param batch_index: batch index
        :param data_container: data container batch
        :return:
        """
        self[-1].summaries.append(data_container.get_ids_summary())
        self[0].put_task(data_container)


class ParallelQueuedFeatureUnion(BaseQueuedPipeline):
    """
    Using :class:`QueueWorker`, run all steps in parallel using QueueWorkers.

    .. seealso::
        :class:`QueueWorker`,
        :class:`BaseQueuedPipeline`,
        :class:`SequentialQueuedPipeline`,
        :class:`QueueJoiner`,
        :class:`Observer`,
        :class:`Observable`
    """

    def get_n_batches(self, data_container):
        """
        Get the number of batches to process by the queue joiner.

        :return:
        """
        return data_container.get_n_batches(self.batch_size) * (len(self) - 1)

    def connect_queued_pipeline(self):
        """
        Connect the queue joiner to all of the queued workers to process data in parallel.

        :return:
        """
        for name, step in self[:-1]:
            step.subscribe_step(self[-1])

    def send_batch_to_queued_pipeline(self, batch_index: int, data_container: DACT):
        """
        Send batches to process to all of the queued workers.

        :param batch_index: batch index
        :param data_container: data container batch
        :return:
        """
        for name, step in self[:-1]:
            queue_joiner: QueueJoiner = self[-1]
            queue_joiner.summaries.append(data_container.get_ids_summary())
            step.put_task(data_container)


class QueueJoiner(ObservableQueueMixin, Joiner):
    """
    Observe the results of the queue worker of type :class:`QueueWorker`.
    Synchronize all of the workers together.

    .. seealso::
        :class:`QueuedPipeline`,
        :class:`Observer`,
        :class:`ListDataContainer`,
        :class:`DataContainer`
    """

    def __init__(self, batch_size, n_batches=None):
        self.n_batches_left_to_do = n_batches
        self.summaries: List[str] = []
        self.result = {}
        Joiner.__init__(self, batch_size=batch_size)
        ObservableQueueMixin.__init__(self, Queue())

    def _teardown(self) -> 'BaseTransformer':
        """
        Properly clean queue, summary ids, and results during teardown.

        :return: teardowned self
        """
        ObservableQueueMixin._teardown(self)
        Joiner._teardown(self)
        self.summaries = []
        self.result: Dict[str, ListDataContainer] = dict()
        return self

    def set_n_batches(self, n_batches):
        self.n_batches_left_to_do = n_batches

    def join(self, original_data_container: DACT) -> DACT:
        """
        Return the accumulated results received by the on next method of this observer.

        :return: transformed data container
        :rtype: DataContainer
        """
        while self.n_batches_left_to_do > 0:
            task: QueuedPipelineTask = self.get_task()
            self.n_batches_left_to_do -= 1
            step_name = task.step_name

            if step_name not in self.result:
                self.result[step_name] = ListDataContainer(
                    ids=[],
                    data_inputs=[],
                    expected_outputs=[]
                )

            self.result[step_name].append_data_container_in_data_inputs(task.data_container)

        data_containers = self._join_all_step_results()
        self.result = {}
        return original_data_container.set_data_inputs(data_containers)

    def _join_all_step_results(self):
        """
        Concatenate all resulting data containers together.

        :return:
        """
        results = []
        for step_name, data_containers in self.result.items():
            self._raise_exception_throwned_by_workers_if_needed(data_containers)
            step_results = self._join_step_results(data_containers)
            results.append(step_results)

        return results

    def _raise_exception_throwned_by_workers_if_needed(self, data_containers: ListDataContainer):
        for dc in data_containers.data_inputs:
            if isinstance(dc, Exception):
                # an exception has been throwned by the worker so reraise it here!
                exception = dc
                raise exception

    def _join_step_results(self, data_containers: ListDataContainer):
        # reorder results by ids of summary
        data_containers.data_inputs.sort(key=lambda dc: self.summaries.index(dc.get_ids_summary()))

        step_results = ListDataContainer.empty()
        for data_container in data_containers.data_inputs:
            step_results.concat(data_container)

        return step_results
