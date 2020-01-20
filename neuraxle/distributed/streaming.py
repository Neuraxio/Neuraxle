"""
Neuraxle steps for streaming data in parallel in the pipeline
===================================================================

Neuraxle steps for streaming data in parallel in the pipeline

..
    Copyright 2019, Neuraxio Inc.

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
from abc import abstractmethod
from multiprocessing import Queue, Lock
from multiprocessing.context import Process
from threading import Thread
from typing import Tuple, List, Union

from neuraxle.base import NamedTupleList, ExecutionContext, BaseStep, MetaStepMixin, NonFittableMixin
from neuraxle.data_container import DataContainer, ListDataContainer
from neuraxle.pipeline import Pipeline, CustomPipelineMixin
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch


class ObservableQueue:
    def __init__(self, queue):
        self.queue = queue
        self.observers = []

    def subscribe(self, queue_worker: 'ObservableQueue') -> 'ObservableQueue':
        self.observers.append(queue_worker.queue)
        return self

    def put(self, value):
        self.queue.put(value)

    def notify(self, value):
        for observer in self.observers:
            observer.put(value)


class QueuedPipelineTask(object):
    def __init__(self, step_name, data_container):
        self.step_name = step_name
        self.data_container = data_container


class QueueWorker(ObservableQueue, MetaStepMixin, BaseStep):
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

    def __init__(self, wrapped: BaseStep, max_size: int, n_workers: int, use_threading: bool):
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)
        ObservableQueue.__init__(self, Queue(maxsize=max_size))

        self.use_threading: bool = use_threading
        self.workers: List[Process] = []
        self.n_workers: int = n_workers
        self.observers: List[Queue] = []

    def start(self, context: ExecutionContext):
        """
        Start multiple processes or threads with the worker function as a target.

        :param context: execution context
        :type context: ExecutionContext
        :return:
        """
        self.workers = []
        for _ in range(self.n_workers):
            if self.use_threading:
                p = Thread(target=self.worker_func, args=(self.queue, context))
            else:
                p = Process(target=self.worker_func, args=(self.queue, context))

            p.daemon = True
            p.start()
            self.workers.append(p)

    def stop(self):
        """
        Stop all of the workers.

        :return:
        """
        if not self.use_threading:
            [w.terminate() for w in self.workers]

        self.workers = []
        self.observers = []

    def worker_func(self, batches_to_process: Queue, context: ExecutionContext):
        """
        Worker function that transforms the items inside the queue of items to process.

        :param batches_to_process: multiprocessing queue
        :param context: execution context
        :type context: ExecutionContext
        :return:
        """
        while True:
            task: QueuedPipelineTask = batches_to_process.get()
            summary_id = task.data_container.summary_id
            data_container = self.handle_transform(task.data_container, context)
            data_container = data_container.set_summary_id(summary_id)
            self.notify(QueuedPipelineTask(step_name=self.wrapped.name, data_container=data_container))


# (step_name, n_workers, step)
# (step_name, n_workers, max_size, step)
# (step_name, step)
QueuedPipelineStepsTuple = Union[Tuple[str, int, BaseStep], Tuple[str, int, int, BaseStep], Tuple[str, BaseStep]]


class BaseQueuedPipeline(NonFittableMixin, CustomPipelineMixin, Pipeline):
    """
    Sub class of :class:`Pipeline`.
    Transform data in many pipeline steps at once in parallel in the pipeline using multiprocessing Queues.

    Example usage :

    .. code-block:: python

        p = QueuedPipeline([
            ('step_a', Identity()),
            ('step_b', Identity()),
        ], n_workers=1, batch_size=10, max_size=10)

        # or

        p = QueuedPipeline([
            ('step_a', 1, Identity()),
            ('step_b', 1, Identity()),
        ], batch_size=10, max_size=10)

        # or

        p = QueuedPipeline([
            ('step_a', 1, 10, Identity()),
            ('step_b', 1, 10, Identity()),
        ], batch_size=10)


    .. seealso::
        :class:`QueueWorker`,
        :class:`QueueJoiner`,
        :class:`CustomPipelineMixin`,
        :class:`Pipeline`
    """

    def __init__(
            self,
            steps: List[QueuedPipelineStepsTuple],
            batch_size,
            n_workers_per_step=None,
            max_size=None,
            data_joiner=None,
            use_threading=False,
            cache_folder=None
    ):
        NonFittableMixin.__init__(self)
        CustomPipelineMixin.__init__(self)

        if data_joiner is None:
            data_joiner = NumpyConcatenateOuterBatch()
        self.data_joiner = data_joiner
        self.max_size = max_size
        self.batch_size = batch_size
        self.n_workers_per_step = n_workers_per_step
        self.use_threading = use_threading

        Pipeline.__init__(self, steps=self._initialize_steps_as_tuple(steps), cache_folder=cache_folder)

    def _initialize_steps_as_tuple(self, steps):
        """
        Wrap each step by a :class:`QueueWorker` to  allow data to flow in many pipeline steps at once in parallel.

        :param steps: (name, n_workers, step)
        :type steps: NameNWorkerStepTupleList
        :return: steps as tuple
        :rtype: NamedTupleList
        """
        steps_as_tuple: NamedTupleList = []
        for step in steps:
            queue_worker = self._create_queue_worker(step)
            steps_as_tuple.append((queue_worker.name, queue_worker))

        return steps_as_tuple

    def _create_queue_worker(self, step: QueuedPipelineStepsTuple):
        name, n_workers, max_size, actual_step = self._get_step_params(step)

        return QueueWorker(
            actual_step,
            n_workers=n_workers,
            use_threading=self.use_threading,
            max_size=max_size
        )

    def _get_step_params(self, step):
        """
        Return all params necessary to create the QueuedPipeline for the given step.

        :param step: tuple
        :type step: QueuedPipelineStepsTupleList

        :return: return name, n_workers, max_size, actual_step
        :rtype: tuple(str, int, int, BaseStep)
        """
        if len(step) == 2:
            name, actual_step = step
            max_size = self.max_size
            n_workers = self.n_workers_per_step
        elif len(step) == 3:
            name, n_workers, actual_step = step
            max_size = self.max_size
        elif len(step) == 4:
            name, n_workers, max_size, actual_step = step
        else:
            raise Exception(
                'Invalid Queued Pipeline Steps Shape. Please use one of the following: (step_name, n_workers, max_size, step), (step_name, n_workers, step), (step_name, step)')

        return name, n_workers, max_size, actual_step

    def _will_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (
            DataContainer, ExecutionContext):
        """
        Start the :class:`QueueWorker` for each step before transforming the data container.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return:
        """
        self.connect_queue_workers()

        return data_container, context

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Transform data container

        :param data_container: data container to transform.
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        """
        data_container_batches = data_container.convolved_1d(stride=self.batch_size, kernel_size=self.batch_size)
        n_batches = data_container.get_n_batches(self.batch_size)

        queue_joiner = QueueJoiner(n_batches=n_batches)
        self.connect_queue_joiner(queue_joiner)

        for i, (name, step) in enumerate(self):
            step.start(context)

        for data_container_batch in data_container_batches:
            self.notify_new_batch_to_process(data_container=data_container_batch, queue_joiner=queue_joiner)

        return queue_joiner.join(original_data_container=data_container)

    def _did_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Stop all of the workers after transform.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        :rtype: DataContainer
        """
        for name, step in self:
            step.stop()

        return self.data_joiner.handle_transform(data_container, context)

    @abstractmethod
    def connect_queue_workers(self):
        raise NotImplementedError()

    @abstractmethod
    def connect_queue_joiner(self, queue_joiner):
        raise NotImplementedError()

    @abstractmethod
    def notify_new_batch_to_process(self, data_container, queue_joiner):
        raise NotImplementedError()


class SequentialQueuedPipeline(BaseQueuedPipeline):
    """
    Using :class:`QueueWorker`, run all steps sequentially even if they are in separate processes or threads.

    .. seealso::
        :class:`QueueWorker`,
        :class:`BaseQueuedPipeline`,
        :class:`ParallelQueuedPipeline`,
        :class:`QueueJoiner`,
        :class:`Observer`,
        :class:`Observable`
    """

    def connect_queue_workers(self):
        for i, (name, step) in enumerate(self):
            if i != 0:
                self[i - 1].subscribe(step)

    def connect_queue_joiner(self, queue_joiner: 'QueueJoiner'):
        self[-1].subscribe(queue_joiner)

    def notify_new_batch_to_process(self, data_container: DataContainer, queue_joiner: 'QueueJoiner'):
        data_container = data_container.set_summary_id(data_container.hash_summary())
        queue_joiner.summary_ids.append(data_container.summary_id)
        self[0].put(QueuedPipelineTask(step_name=self[0].name, data_container=data_container))


class ParallelQueuedPipeline(BaseQueuedPipeline):
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

    def connect_queue_workers(self):
        # nothing to do here, queue workers don't listen to each other in a ParallelQueuedPipeline
        pass

    def connect_queue_joiner(self, queue_joiner: 'QueueJoiner'):
        for i, (name, step) in enumerate(self):
            step.subscribe(queue_joiner)

    def notify_new_batch_to_process(self, data_container, queue_joiner: 'QueueJoiner'):
        for i, (name, step) in enumerate(self):
            data_container = data_container.set_summary_id(data_container.hash_summary())
            queue_joiner.summary_ids.append(data_container.summary_id)
            step.put(QueuedPipelineTask(step_name=step.name, data_container=data_container))


class QueueJoiner(ObservableQueue):
    """
    Observe the results of the queue worker of type :class:`QueueWorker`.
    Synchronize all of the workers together.

    .. seealso::
        :class:`QueuedPipeline`,
        :class:`Observer`,
        :class:`ListDataContainer`,
        :class:`DataContainer`
    """

    def __init__(self, n_batches):
        self.mutex_processing_in_progress = Lock()
        self.mutex_processing_in_progress.acquire()
        self.n_batches_left_to_do = n_batches
        self.summary_ids = []
        self.result = {}
        ObservableQueue.__init__(self, Queue())

    def join(self, original_data_container: DataContainer) -> DataContainer:
        """
        Return the accumulated results received by the on next method of this observer.

        :return: transformed data container
        :rtype: DataContainer
        """
        while self.n_batches_left_to_do > 0:
            task: QueuedPipelineTask = self.queue.get()
            self.n_batches_left_to_do -= 1

            if task.step_name not in self.result:
                self.result[task.step_name] = ListDataContainer(
                    current_ids=[],
                    data_inputs=[],
                    expected_outputs=[],
                    summary_id=task.data_container.summary_id
                )

            self.result[task.step_name].append_data_container(task.data_container)

        data_containers = self._join_all_step_results()
        return original_data_container.set_data_inputs(data_containers)

    def _join_all_step_results(self):
        """
        Concatenate all resulting data containers together.

        :return:
        """
        results = ListDataContainer.empty()
        for step_name, data_containers in self.result.items():
            step_results = self._join_step_results(data_containers)
            results.append_data_container(step_results)

        return results

    def _join_step_results(self, data_containers):
        # reorder results by summary id
        list(data_containers.data_inputs).sort(key=lambda dc: self.summary_ids.index(dc.summary_id))

        step_results = ListDataContainer.empty()
        for data_container in data_containers.data_inputs:
            data_container = data_container.set_summary_id(data_containers.data_inputs[-1].summary_id)
            step_results.concat(data_container)

        return step_results
