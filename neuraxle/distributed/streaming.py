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

from neuraxle.base import NamedTupleList, ExecutionContext, BaseStep, MetaStepMixin
from neuraxle.data_container import DataContainer, ListDataContainer
from neuraxle.pipeline import Pipeline, CustomPipelineMixin


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


class Observable:
    """
    Observable class that notifies observers of type :class:`Observer`.
    """
    def __init__(self):
        self.observers: List[Observer] = []

    def subscribe(self, observer: Observer):
        """
        Add observer to the subscribed observers list.

        :param observer: observer
        :type observer: Observer
        :return:
        """
        self.observers.append(observer)

    def on_next(self, value):
        """
        Notify all of the observers.

        :param value:
        :return:
        """
        for observer in self.observers:
            observer.on_next(value)


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
            Observable.on_next(self, data_container)


# [step_name, n_workers, step]
# [step_name, n_workers, max_size, step]
# [step_name, step]
QueuedPipelineStepsTupleList = Union[List[Tuple[str, int, BaseStep]], List[Tuple[str, int, int, BaseStep]], List[Tuple[str, BaseStep]]]


class QueuedPipeline(CustomPipelineMixin, Pipeline):
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
    def __init__(self, steps: QueuedPipelineStepsTupleList, batch_size, max_size, use_threading=False, cache_folder=None):
        CustomPipelineMixin.__init__(self)

        self.max_size = max_size
        self.batch_size = batch_size
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
        for name, n_workers, step in steps:
            wrapped_step = QueueWorker(
                step,
                n_workers=n_workers,
                use_threading=self.use_threading,
                max_size=self.max_size
            )

            should_subscribe_to_previous_step = len(steps_as_tuple) > 0 and len(steps_as_tuple) != len(steps) - 1

            if should_subscribe_to_previous_step:
                _, previous_step = steps_as_tuple[-1]
                wrapped_step.subscribe(previous_step)

            steps_as_tuple.append((name, wrapped_step))

        return steps_as_tuple

    def _will_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Start the :class:`QueueWorker` for each step before transforming the data container.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return:
        """
        for name, step in self:
            step.start(context)

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
        n_batches = data_container.get_n_baches(self.batch_size)
        queue_joiner = QueueJoiner(n_batches=n_batches)

        batches_observable = Observable()
        batches_observable.subscribe(self[0])
        self[-1].subscribe(queue_joiner)

        for data_container_batch in data_container_batches:
            batches_observable.on_next(data_container_batch)

        return queue_joiner.join()

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

        return data_container


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
        self.result = ListDataContainer(
            current_ids=[],
            data_inputs=[],
            expected_outputs=[],
            summary_id=None
        )

    def on_next(self, value):
        """
        Receive the final results of a batch processed by the queued pipeline.
        Releases the mutex_processing_in_progress if there is no batches left to process.

        :param value: transformed data container batch
        :type value: DataContainer
        :return:
        """
        self.n_batches_left_to_do -= 1

        self.result.concat(value)

        if self.n_batches_left_to_do == 0:
            self.mutex_processing_in_progress.release()

    def join(self) -> DataContainer:
        """
        Return the accumulated results received by the on next method of this observer.

        :return: transformed data container
        :rtype: DataContainer
        """
        self.mutex_processing_in_progress.acquire()
        return self.result
