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
from abc import abstractmethod, ABC
from multiprocessing import Queue, Lock
from multiprocessing.context import Process
from threading import Thread
from typing import Tuple, List, Union, Callable

from neuraxle.base import NamedTupleList, ExecutionContext, BaseStep, MetaStepMixin, NonFittableMixin
from neuraxle.data_container import DataContainer, ListDataContainer
from neuraxle.pipeline import Pipeline, CustomPipelineMixin
from neuraxle.steps.numpy import NumpyConcatenateOuterBatch


class Observer:
    """
    Observer class that listens to :class:`Observable` events.

    .. seealso::
        :class:`Observable`
    """

    @abstractmethod
    def on_next(self, value):
        """
        Method called by the observables to notifiy the observers.

        :param value:
        :return:
        """
        pass


class Subject(Observer):
    def __init__(self, on_next_fun: Callable = None):
        self.on_next_fun = on_next_fun
        self.observable = Observable()

    def subscribe(self, observer):
        self.observable = self.observable.subscribe(observer)

    def on_next(self, value):
        if self.on_next_fun is not None:
            value = self.on_next_fun(value)
        self.observable.on_next(value)


class Observable:
    """
    Observable class that notifies observers of type :class:`Observer`.
    """

    def __init__(self):
        self.observers: List[Observer] = []

    def subscribe(self, observer: Observer) -> 'Observable':
        """
        Add observer to the subscribed observers list.

        :param observer: observer
        :type observer: Observer
        :return:
        """
        self.observers.append(observer)
        return self

    def on_next(self, value):
        """
        Notify all of the observers.

        :param value:
        :return:
        """
        for observer in self.observers:
            observer.on_next(value)

    def map(self, map_fun: Callable):
        mappped_observable = Subject(on_next_fun=map_fun)
        self.subscribe(mappped_observable)

        return mappped_observable

    def unsubscribe_all(self):
        self.observers = []


class QueueWorker(Observer, Observable, MetaStepMixin, BaseStep):
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
        Observer.__init__(self)
        Observable.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)

        self.use_threading: bool = use_threading
        self.workers: List[Process] = []
        self.batches_to_process: Queue = Queue(maxsize=max_size)
        self.n_workers: int = n_workers

    def on_next(self, value):
        """
        Add batch to process when the observer receives a value.

        :param value: data container to process
        :type value: DataContainer
        :return:
        """
        self.batches_to_process.put(value)

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
                p = Thread(target=self.worker_func, args=(self.batches_to_process, context))
            else:
                p = Process(target=self.worker_func, args=(self.batches_to_process, context))

            p.daemon = True
            p.start()
            self.workers.append(p)

    def stop(self):
        """
        Stop all of the workers.

        :return:
        """
        if not self.use_threading:
            [w.kill() for w in self.workers]
        self.workers = []
        self.unsubscribe_all()

    def worker_func(self, batches_to_process, context: ExecutionContext):
        """
        Worker function that transforms the items inside the queue of items to process.

        :param batches_to_process: multiprocessing queue
        :param context: execution context
        :type context: ExecutionContext
        :return:
        """
        while True:
            data_container = batches_to_process.get()
            data_container = self.handle_transform(data_container, context)
            Observable.on_next(self, (self.wrapped.name, data_container))


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

    def _will_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> (DataContainer, ExecutionContext):
        """
        Start the :class:`QueueWorker` for each step before transforming the data container.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return:
        """
        self.connect_queue_workers()
        for i, (name, step) in enumerate(self):
            step.start(context)

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

        batches_observable = Observable()
        self.connect_batches_observable(batches_observable)

        queue_joiner = QueueJoiner(n_batches=n_batches)
        self.connect_queue_joiner(queue_joiner)

        for data_container_batch in data_container_batches:
            batches_observable.on_next(data_container_batch)

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

        return self.data_joiner.join(data_container)

    def map_result(self, value):
        return value[1]

    @abstractmethod
    def connect_queue_workers(self):
        raise NotImplementedError()

    @abstractmethod
    def connect_queue_joiner(self, queue_joiner):
        raise NotImplementedError()

    @abstractmethod
    def connect_batches_observable(self, batches_observable):
        raise NotImplementedError()


class SequentialQueuedPipeline(BaseQueuedPipeline):
    """
    Run all steps sequentially using QueueWorkers.
    """
    def connect_queue_workers(self):
        for i, (name, step) in enumerate(self):
            if i != 0:
                self[i - 1].map(self.map_result).subscribe(step)

    def connect_queue_joiner(self, queue_joiner):
        self[-1].subscribe(queue_joiner)

    def connect_batches_observable(self, batches_observable):
        batches_observable.subscribe(self[0])


class ParallelQueuedPipeline(BaseQueuedPipeline):
    """
    Run all steps in parallel using QueueWorkers.
    """
    def connect_queue_workers(self):
        # nothing to do here, queue workers don't listen to each other in a ParallelQueuedPipeline
        pass

    def connect_queue_joiner(self, queue_joiner):
        for i, (name, step) in enumerate(self):
            step.subscribe(queue_joiner)

    def connect_batches_observable(self, batches_observable):
        for i, (name, step) in enumerate(self):
            batches_observable.subscribe(step)


class QueueJoiner(Observer):
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
        self.result = {}

    def on_next(self, value):
        """
        Receive the final results of a batch processed by the queued pipeline.
        Releases the mutex_processing_in_progress if there is no batches left to process.

        :param value: transformed data container batch
        :type value: DataContainer
        :return:
        """
        step_name, result = value

        self.n_batches_left_to_do -= 1

        if step_name not in result:
            self.result[step_name] = ListDataContainer(
                current_ids=[],
                data_inputs=[],
                expected_outputs=[],
                summary_id=None
            )

        result[step_name].concat(value)

        if self.n_batches_left_to_do == 0:
            self.mutex_processing_in_progress.release()

    def join(self, original_data_container: DataContainer) -> DataContainer:
        """
        Return the accumulated results received by the on next method of this observer.

        :return: transformed data container
        :rtype: DataContainer
        """
        self.mutex_processing_in_progress.acquire()
        data_containers = []
        for data_container in self.result.values():
            data_containers.append(data_container)

        return original_data_container.set_data_inputs(data_containers)
