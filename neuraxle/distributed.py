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
import os
from abc import ABC, abstractmethod
from queue import Queue
from typing import List, Iterable

import numpy as np
import requests
from flask import jsonify
from werkzeug.utils import secure_filename

from neuraxle.api.flask import RestAPICaller
from neuraxle.base import MetaStepMixin, BaseStep, NonFittableMixin, ExecutionContext
from neuraxle.data_container import DataContainer
from neuraxle.steps.numpy import NumpyConcatenateInnerFeatures


class BaseClusteringScheduler(ABC):
    """
    Base class for a clustering scheduler that distributes task to multiple workers.
    Used by the :class:`ClusteringWrapper` for distributed processing.

    .. seealso::
        :class:`RestApiScheduler`,
        :class:`ClusteringWrapper`,
        :class:`DataContainer`,
        :class:`ExecutionContext`
    """
    def __init__(self):
        self.started = False

    @abstractmethod
    def spawn(self, context: ExecutionContext):
        pass

    @abstractmethod
    def dispatch(self, batches: Iterable[DataContainer], context: ExecutionContext) -> List[DataContainer]:
        pass

    @abstractmethod
    def kill(self, context: ExecutionContext):
        pass


class RestApiScheduler(BaseClusteringScheduler):
    """
    Clustering scheduler that calls rest api endpoints for distributed processing.

    Please refer to the full example in :class:`ClusteringWrapper` for further details.

    .. seealso::
        :class:`BaseClusteringScheduler`,
        :class:`ClusteringWrapper`,
        :class:`ExecutionContext`,
        :class:`RestApiCaller`,
    """
    def __init__(self, hosts: List[str]):
        super().__init__()
        self.hosts = hosts
        self.workers = Queue()

    def dispatch(self, batches: Iterable[DataContainer], context: ExecutionContext) -> List[DataContainer]:
        """
        Dispatch tasks to the workers using rest api calls.

        :param batches: an iterable list of data containers
        :param context: execution context
        :type context: ExecutionContext
        :return: list of data container
        :rtype: List[DataContainer]
        """
        results = []

        for batch in batches:
            worker = self.workers.get()
            results.append(worker.handle_transform(batch, context))
            self.workers.put(worker)

        return results

    def kill(self, context: ExecutionContext):
        # TODO: what is there to kill here ?
        pass

    def spawn(self, context: ExecutionContext):
        """
        Spawn a worker for each host.

        1) Send the full pipeline dump files to the workers using the rest api endpoints in self.hosts
        2) Add an :class:`ApiCaller` step in the queue of workers for each hosts.

        :param context: execution context
        :return:
        """
        self.started = True
        for host in self.hosts:
            saved_step_files = context.get_all_saved_step_files()
            files = {path.replace(str(context.root) + '/', ''): open(file_path, 'rb') for path, file_path in saved_step_files}
            r = requests.post(host, files=files)

            for f in files.values():
                f.close()

            if r.status_code == 201:
                self.workers.put(RestAPICaller(host))


class ClusteringWrapper(MetaStepMixin, BaseStep):
    """
    Wrapper step for distributed processing. A :class:`BaseClusteringScheduler`, distributes the transform method to multiple workers.

    Example usage :

    .. code-block:: python

        p = Pipeline([
            ClusteringWrapper(
                Pipeline([..]),
                scheduler=RestApiScheduler(['http://127.0.0.1:5000/pipeline']),
                joiner=NumpyConcatenateOuterBatch(),
                n_jobs=10,
                batch_size=10
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
        wrapped_context = context.push(self.wrapped)
        wrapped_context = wrapped_context.push(wrapped_context.save_last())

        self.scheduler.spawn(wrapped_context)

        # create batches to transform
        batch_size = self._get_batch_size(data_container)
        data_container_batches = data_container.convolved_1d(stride=batch_size, kernel_size=batch_size)

        # dispatch batches to transform
        data_containers = self.scheduler.dispatch(data_container_batches, wrapped_context)

        # kill all the workers
        self.scheduler.kill(wrapped_context)

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


class RestWorker:
    """
    Worker class for the :class:`RestApiScheduler`.


    Easily start a flask server for a rest api worker :

    .. code-block:: python

        worker = RestWorker(host='127.0.0.1', port=5000)
        worker.start(upload_folder='/home/neuraxle/Documents/cluster')


    .. seealso::
        :class:`BaseClusteringScheduler`,
        :class:`LocalScheduler`,
        :class:`ExternalScheduler`,
        :class:`MetaStepMixin`,
        :class:`BaseStep`,
        :class:`DataContainer`
    """
    def __init__(self, ressource_name='pipeline', host: str = None, port: int = 5000):
        self.ressource_name = ressource_name
        self.host = host
        self.port = port

    def get_app(self):
        """
        :return: a Flask app (as given by `app = Flask(__name__)` and then configured).
        """
        from flask import Flask, request
        from flask_restful import Api, Resource

        app = Flask(__name__)
        api = Api(app)

        class PipelineRessource(Resource):
            def post(self):
                """
                Post route to save the full pipeline dump locally.
                Receives the full dump for each pipeline steps inside the files attribute.

                :return: response
                """
                if len(request.files) == 0:
                    response = jsonify({'message': 'No file part in the request'})
                    response.status_code = 400
                    return response

                for file in request.files:
                    filename = secure_filename(request.files[file].filename)
                    upload_path = os.path.join(app.config['UPLOAD_FOLDER'], file.replace(filename, ''))

                    if not os.path.exists(upload_path):
                        os.makedirs(upload_path)

                    file_path = os.path.join(upload_path, filename)
                    request.files[file].save(file_path)

                response = jsonify({'message': 'Files successfully uploaded'})
                response.status_code = 201

                return response

            def get(self):
                """
                Get route to transform the data container received in the body :

                .. code-block:: python

                    {
                        current_ids: [...],
                        data_inputs: [...],
                        expected_outputs: [...],
                        summary_id: ...
                    }

                :return: transformed data container
                """
                context = ExecutionContext(app.config['UPLOAD_FOLDER'])
                body = request.get_json()
                pipeline = context.load(body['path'])

                data_container = pipeline.handle_transform(
                    DataContainer(
                        current_ids=body['current_ids'],
                        data_inputs=body['data_inputs'],
                        expected_outputs=body['expected_outputs'],
                        summary_id=body['summary_id'],
                    ),
                    context
                )

                response = jsonify({
                    'current_ids': np.array(data_container.current_ids).tolist(),
                    'data_inputs': np.array(data_container.data_inputs).tolist(),
                    'expected_outputs': np.array(data_container.expected_outputs).tolist(),
                    'summary_id': data_container.summary_id
                })
                response.status_code = 200

                return response

        api.add_resource(PipelineRessource, '/' + self.ressource_name)

        return app

    def start(self, upload_folder):
        """
        Start the worker with the given upload folder.

        :param upload_folder: upload folder
        :type upload_folder: str
        :return:
        """
        app = self.get_app()
        app.config['UPLOAD_FOLDER'] = upload_folder
        app.run(host=self.host, debug=False, port=self.port)

