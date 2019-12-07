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

import numpy as np
from fs.memoryfs import MemoryFS
from joblib import dump, Parallel, delayed, load

from neuraxle.base import BaseStep, MetaStepMixin, NonFittableMixin, ExecutionContext, Identity, BaseSaver, \
    DEFAULT_CACHE_FOLDER, ExecutionMode
from neuraxle.data_container import DataContainer


class MemoryFSJoblibSaver(BaseSaver):
    def __init__(self, memory_file_system):
        self.memory_file_system = memory_file_system

    def save_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        step_path = self._create_step_path(context, step)

        self.memory_file_system.makedir(step_path)

        with self.memory_file_system.open(step_path, 'w+') as file:
            dump(file, step_path)

        return step

    def can_load(self, step: 'BaseStep', context: 'ExecutionContext'):
        return self.memory_file_system.exists(self._create_step_path(context, step))

    def load_step(self, step: 'BaseStep', context: 'ExecutionContext') -> 'BaseStep':
        step_path = self._create_step_path(context, step)

        with self.memory_file_system.open(step_path, 'r') as file:
            return load(file)

    def _create_step_path(self, context, step):
        return os.path.join(context.get_path(), '{0}.joblib'.format(step.name))

class MemoryFSExecutionContext(ExecutionContext):
    def __init__(self,
        memory_file_system: MemoryFS,
        root: str = DEFAULT_CACHE_FOLDER,
        execution_mode: ExecutionMode = None,
        stripped_saver: BaseSaver = None,
        parents=None
    ):
        ExecutionContext.__init__(
            self,
            root=root,
            execution_mode=execution_mode,
            stripped_saver=stripped_saver,
            parents=parents
        )
        self.memory_file_system = memory_file_system

    def mkdir(self):
        path = self.get_path()
        parts = path.split(os.sep)

        dir_to_create = ''
        for i in range(len(parts)):
            dir_to_create += parts[i] + os.sep
            if not self.memory_file_system.exists(dir_to_create):
                self.memory_file_system.makedir(dir_to_create)

    def push(self, step: 'BaseStep'):
        return MemoryFSExecutionContext(
            memory_file_system=self.memory_file_system,
            root=self.root,
            execution_mode=self.execution_mode,
            parents=self.parents + [step],
            stripped_saver=self.stripped_saver
        )

    def get_path(self):
        parents_with_path = [p.name for p in self.parents]
        return os.path.join(*parents_with_path)


class SaverParallelTransform(NonFittableMixin, BaseStep):
    """
    Use savers to parallelize steps transformations to avoid python limitations when importing external librairies.
    Dispatching technique class to abstract the workers.

    .. seealso::
        :func:`~NonFittableMixin`,
        :func:`~MetaStepMixin`,
        :class:`BaseStep`
    """
    def __init__(self, wrapped: BaseStep, n_jobs: int = None):
        MetaStepMixin.__init__(self, wrapped)
        BaseStep.__init__(self)
        self.n_jobs = n_jobs

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        with MemoryFS() as memory_file_system:
            shared_memory_execution_context = MemoryFSExecutionContext(
                memory_file_system=memory_file_system,
                root=memory_file_system.root,
                parents=context.parents,
                stripped_saver=MemoryFSJoblibSaver(memory_file_system)
            )
            # shared_memory_execution_context.save(full_dump=True)
            execution_context_path = shared_memory_execution_context.get_path()

            # http://lagrange.univ-lyon1.fr/docs/numpy/1.11.0/reference/generated/numpy.memmap.html#numpy-memmap
            # https://joblib.readthedocs.io/en/latest/auto_examples/parallel_memmap.html#sphx-glr-auto-examples-parallel-memmap-py

            with memory_file_system.open('data_inputs', 'w') as file:
                memory_map_data_inputs = np.memmap(
                    file,
                    dtype=data_container.data_inputs.dtype,
                    mode='w+',
                    shape=data_container.data_inputs.shape
                )
                memory_map_data_inputs[:] = data_container.data_inputs[:]

            with memory_file_system.open('expected_outputs', 'w') as file:
                memory_map_expected_outputs = np.memmap(
                    file,
                    dtype=data_container.expected_outputs.dtype,
                    mode='w+',
                    shape=data_container.expected_outputs.shape
                )
                memory_map_expected_outputs[:] = data_container.expected_outputs[:]

            # todo create a data container for each job, and create a summary id for each job

            # todo loop through summary id for each jobs
            outputs = Parallel(
                n_jobs=-1,
                backend='multiprocessing'
            )(delayed(self.receive)(execution_context_path, data_container.copy()))

        return data_container.set_data_inputs(outputs)

    def transform(self, data_inputs):
        raise Exception('Transform method not supported by SharedMemoryDispatcher. Please use this step inside a pipeline'.format(repr(self)))

    def receive(self, execution_context_path, data_inputs, expected_outputs=None):
        step_names = execution_context_path.split(os.sep)

        parents = [Identity(name) for name in step_names]
        context = ExecutionContext(parents=parents)
        step = context.load(self.name)

        return step.transform(data_inputs, expected_outputs)
