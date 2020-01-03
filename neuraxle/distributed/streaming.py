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
from typing import Tuple, List

from neuraxle.base import NamedTupleList, ExecutionContext, BaseStep, MetaStepMixin
from neuraxle.data_container import DataContainer, ListDataContainer
from neuraxle.pipeline import Pipeline, CustomPipelineMixin


class Observer:
    @abstractmethod
    def on_next(self, value):
        pass


class Observable:
    def __init__(self):
        self.observers = []

    def subscribe(self, observer: Observer):
        self.observers.append(observer)

    def on_next(self, value):
        for observer in self.observers:
            observer.on_next(value)


class QueueWorker(Observer, Observable, MetaStepMixin, BaseStep):
    def __init__(self, wrapped: BaseStep, n_workers: int, use_threading: bool):
        Observer.__init__(self)
        Observable.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)

        self.use_threading: bool = use_threading
        self.worker_processes: List[Process] = []
        self.batches_to_process: Queue = Queue(n_workers)
        self.n_workers: int = n_workers

    def on_next(self, value):
        self.batches_to_process.put(value)

    def start(self, context):
        worker_processes = []
        for _ in range(self.n_workers):
            if self.use_threading:
                p = Thread(target=self.worker_func, args=(self.batches_to_process, context))
            else:
                p = Process(target=self.worker_func, args=(self.batches_to_process, context))

            p.daemon = True
            p.start()
            worker_processes.append(p)

    def stop(self):
        if not self.use_threading:
            [w.kill() for w in self.worker_processes]
        self.worker_processes = []

    def worker_func(self, batches_to_process, context: ExecutionContext):
        while True:
            data_container = batches_to_process.get()
            data_container = self.handle_transform(data_container, context)
            Observable.on_next(self, data_container)


NameNWorkerStepTupleList = List[Tuple[str, int, BaseStep]]


class QueuedPipeline(CustomPipelineMixin, Pipeline):
    def __init__(self, steps: NameNWorkerStepTupleList, max_batches, batch_size, use_threading=False, cache_folder=None):
        CustomPipelineMixin.__init__(self)

        self.max_batches = max_batches
        self.batch_size = batch_size
        self.use_threading = use_threading

        Pipeline.__init__(self, steps=self._initialize_steps_as_tuple(steps), cache_folder=cache_folder)

    def _initialize_steps_as_tuple(self, steps):
        if not isinstance(steps, tuple):
            steps = [(step.name, step) for step in steps]

        steps_as_tuple: NamedTupleList = []
        for name, n_workers, step in steps:
            wrapped_step = QueueWorker(step, n_workers=n_workers, use_threading=self.use_threading)
            if len(steps_as_tuple) > 0:
                previous_step: Observer = steps_as_tuple[-1][1]
                wrapped_step.subscribe(previous_step)

            steps_as_tuple.append((name, wrapped_step))

        return steps_as_tuple

    def _will_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        for name, step in self:
            step.start(context)

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Transform data container

        :param data_container: data container to transform.
        :param context: execution context
        :return: data container
        """
        data_container_batches = data_container.convolved_1d(stride=self.batch_size, kernel_size=self.batch_size)
        n_batches = data_container.get_n_baches(self.batch_size)
        queue_joiner = QueueJoiner(n_batches=n_batches, max_batches=self.max_batches)

        batches_observable = Observable()
        batches_observable.subscribe(self[0])
        self[-1].subscribe(queue_joiner)

        for data_container_batch in data_container_batches:
            queue_joiner.add_batch_in_progress()
            batches_observable.on_next(data_container_batch)

        return queue_joiner.join()

    def _did_transform(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        for name, step in self:
            step.stop()

        return data_container


class QueueJoiner(Observer):
    def __init__(self, n_batches, max_batches):
        self.mutex = Lock()
        self.mutex.acquire()
        self.max_batches = max_batches
        self.n_batches_left_to_do = n_batches
        self.batches_in_progress = 0
        self.result = ListDataContainer(current_ids=[], data_inputs=[], expected_outputs=[], summary_id=None)

    def add_batch_in_progress(self):
        self.batches_in_progress += 1

    def on_next(self, value):
        self.n_batches_left_to_do -= 1
        self.batches_in_progress -= 1

        self.result.concat(value)

        if self.n_batches_left_to_do == 0:
            self.mutex.release()

    def join(self) -> DataContainer:
        self.mutex.acquire()
        return self.result
