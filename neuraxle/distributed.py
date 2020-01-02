"""
Neuraxle steps for distributed computing in the cloud
======================================================

Neuraxle Steps for distributed computing in the cloud

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
import math
from abc import ABC, abstractmethod
from queue import Queue
from typing import List, Iterable

from neuraxle.base import MetaStepMixin, BaseStep, NonFittableMixin, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures


class BaseClusteringScheduler(ABC):
    def __init__(self):
        self.started = False

    @abstractmethod
    def spawn(self, context: ExecutionContext):
        pass

    @abstractmethod
    def dispatch(self, batches: Iterable[DataContainer]):
        pass

    @abstractmethod
    def get_results(self) -> DataContainer:
        pass

    @abstractmethod
    def kill(self, context: ExecutionContext):
        pass


class LocalScheduler(BaseClusteringScheduler):
    def dispatch(self, batches: Iterable[DataContainer]):
        workers = Queue()

        self.results = []

        for batch in batches:
            worker = workers.get()
            worker.transform(batch)
            workers.put(worker)

    def get_results(self) -> List[DataContainer]:
        return self.results

    def kill(self, context: ExecutionContext):
        pass

    def spawn(self, context: ExecutionContext):
        self.started = True


class LocalWorker:
    def start(self):
        # start api that saves a pipeline dump ?
        app = FlaskRestApiWrapper(
            json_decoder=CustomJSONDecoderFor2DArray(),
            wrapped=Pipeline(...),
            json_encoder=CustomJSONEncoderOfOutputs(),
        ).get_app()

        # start api/socket that transforms the data container ?
        app = FlaskRestApiWrapper(
            json_decoder=CustomJSONDecoderFor2DArray(),
            wrapped=Pipeline(...),
            json_encoder=CustomJSONEncoderOfOutputs(),
        ).get_app()
        # add a socket send step at the end of the pipeline

        app.run(debug=False, port=5000)


class ExternalScheduler(BaseClusteringScheduler):
    def __init__(self, hosts: List[str]):
        super().__init__()
        self.hosts = hosts

    def spawn(self, context: ExecutionContext):
        self.started = True
        pass

    def dispatch(self, batches: Iterable[DataContainer]):
        pass

    def get_results(self) -> DataContainer:
        pass

    def kill(self, context: ExecutionContext):
        pass


class ClusteringWrapper(MetaStepMixin, BaseStep):
    """
    Wrapper step for distributed processing. A :class:`BaseClusteringScheduler`, distributes the transform method to multiple workers.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            ClusteringWrapper(
                Pipeline([..]),
                scheduler=LocalScheduler(),
                joiner=NumpyConcatenateInnerFeatures(),
                n_jobs=10,
                batch_size=100
            )
        ])

    .. seealso::
        :class:`BaseClusteringScheduler`,
        :class:`LocalScheduler`,
        :class:`ExternalScheduler`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`,
        :class:`DataContainer`
    """

    def __init__(
            self,
            wrapped: BaseStep,
            scheduler: BaseClusteringScheduler,
            joiner: NonFittableMixin = None,
            n_jobs: int = 10,
            batch_size=None
    ):
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)
        self.scheduler = scheduler
        if joiner is None:
            joiner = NumpyConcatenateInnerFeatures()
        self.joiner = joiner

        self.batch_size = batch_size
        self.n_jobs = n_jobs

    def _fit_transform_data_container(self, data_container, context):
        raise Exception('fit not supported by ClusteringWrapper')

    def _fit_data_container(self, data_container, context):
        raise Exception('fit not supported by ClusteringWrapper')

    def _transform_data_container(self, data_container, context):
        """
        Transform data container using distributed processing.

        :param data_container:
        :param context:
        :return:
        """
        context.save_last()

        self.scheduler.spawn(context)

        # create batches to transform
        batch_size = self._get_batch_size(data_container)
        data_container_batches = data_container.convolved_1d(stride=batch_size, kernel_size=batch_size)

        # dispatch batches to transform
        self.scheduler.dispatch(data_container_batches)

        # wait for the tasks to complete and retrieve the results.
        data_containers = self.scheduler.get_results()

        # kill all the workers
        self.scheduler.kill(context)

        # return the resulting data containers for each task
        return DataContainer(
            summary_id=data_container.summary_id,
            current_ids=data_container.current_ids,
            data_inputs=data_containers,
            expected_outputs=data_container.expected_outputs
        )

    def _did_transform(self, data_container, context):
        """
        Join results together using the joiner.

        :param data_container:
        :param context:
        :return:
        """
        return self.joiner.handle_transform(data_container, context)

    def _get_batch_size(self, data_container: DataContainer) -> int:
        """
        Get batch size.

        :param data_container: data container
        :type data_container: DataContainer
        :return: batch_size
        :rtype: int
        """
        if self.batch_size is None:
            batch_size = math.ceil(len(data_container) / self.n_jobs)
        else:
            batch_size = self.batch_size
        return batch_size
