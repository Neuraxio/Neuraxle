"""
Neuraxle steps for parallel processing
================================================

Neuraxle Steps for parallel processing

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
import os
from abc import abstractmethod

import numpy as np
from joblib import Parallel, delayed

from neuraxle.base import BaseStep, MetaStepMixin, NonFittableMixin, ExecutionContext, Identity
from neuraxle.data_container import DataContainer



class DispacherStep(BaseStep):
    def __init__(self):
        BaseStep.__init__(self)

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        pass

    @abstractmethod
    def dispatch(self, data_container: DataContainer, context: ExecutionContext):
        raise NotImplementedError()

    def set_step_name(self, name: str):
        self.name = name


class SharedMemoryDispatcherStep(DispacherStep):
    def dispatch(self, data_container: DataContainer, context: ExecutionContext):
        data_inputs = np.array(data_container.data_inputs)
        expected_outputs = np.array(data_container.expected_outputs)

        shared_memory_data_inputs = shared_memory.SharedMemory(create=True, size=data_inputs.nbytes)
        shared_memory_expected_outputs = shared_memory.SharedMemory(create=True, size=expected_outputs.npbytes)
        shm_a = shared_memory.SharedMemory(create=True, size=10)

        pass


class MultiprocessingDispatcher(DispacherStep):
    def dispatch(self, data_container: DataContainer, context: ExecutionContext):
        return Parallel(n_jobs=-1)(
            delayed(self.receive)(di, eo)
            for di, eo in zip(context.get_path(), data_container.data_inputs, data_container.expected_outputs)
        )

    def receive(self, execution_context_path, data_inputs, expected_outputs=None):
        step_names = execution_context_path.split(os.sep)

        parents = [Identity(name) for name in step_names]
        context = ExecutionContext(parents=parents)
        step = context.load(self.name)

        return step.transform(data_inputs, expected_outputs)


class SaverParallelTransform(NonFittableMixin, MetaStepMixin, BaseStep):
    """
    Use savers to parallelize steps transformations to avoid python limitations when importing external librairies.
    Dispatching technique class to abstract the workers.

    .. seealso::
        :func:`~NonFittableMixin`,
        :func:`~MetaStepMixin`,
        :class:`BaseStep`
    """

    def __init__(
            self,
            wrapped: BaseStep,
            dispatcher: DispacherStep
    ):
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)

        self.dispatcher = dispatcher
        self.dispatcher.set_step_name(self.wrapped.name)

    def _will_transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        self.wrapped.save(context, full_dump=True)

    def _transform_data_container(self, data_container, context):
        data_container = self.dispatcher.handle_transform(data_container, context)
