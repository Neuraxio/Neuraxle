"""
Neuraxle's Checkpoint Classes
====================================
The checkpoint classes used by the checkpoint pipeline runner
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
import pickle
from abc import abstractmethod
from typing import Any

from neuraxle.base import ResumableStepMixin, BaseStep, DataContainer

DEFAULT_CACHE_FOLDER = os.path.join(os.getcwd(), 'cache')


class BaseCheckpointStep(ResumableStepMixin, BaseStep):
    """
    Base class for a checkpoint step that can persists the received data inputs, and expected_outputs
    to eventually be able to load them using the checkpoint pipeline runner.
    """

    def __init__(self, force_checkpoint_name: str = None):
        ResumableStepMixin.__init__(self)
        BaseStep.__init__(self)
        self.force_checkpoint_name = force_checkpoint_name

    def handle_transform(self, data_container: DataContainer) -> Any:
        self.save_checkpoint(data_container)
        return data_container

    def handle_fit_transform(self, data_container: DataContainer) -> ('BaseStep', Any):
        self.save_checkpoint(data_container)
        return self, data_container

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseCheckpointStep':
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run
        :param expected_outputs: initial expected outputs of pipeline to load checkpoint from
        :param data_inputs: data inputs to save
        :return: self
        """
        return self

    def transform(self, data_inputs):
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run
        :param data_inputs: data inputs to save
        :return: data_inputs
        """
        return data_inputs

    @abstractmethod
    def set_checkpoint_path(self, path):
        """
        Set checkpoint Path
        :param path: checkpoint path
        """
        raise NotImplementedError()

    @abstractmethod
    def read_checkpoint(self, data_container: DataContainer) -> DataContainer:
        """
        Read checkpoint data to get the data inputs and expected output.
        :param data_container: data inputs to save
        :return: data_inputs_checkpoint
        """
        raise NotImplementedError()

    @abstractmethod
    def save_checkpoint(self, data_container: DataContainer):
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run
        :param ids: data inputs ids
        :param data_container: data inputs to save
        :return:
        """
        raise NotImplementedError()


class PickleCheckpointStep(BaseCheckpointStep):
    """
    Create pickles for a checkpoint step data inputs, and expected_outputs
    to eventually be able to load them using the checkpoint pipeline runner.
    """

    def __init__(self, force_checkpoint_name: str = None, cache_folder: str = DEFAULT_CACHE_FOLDER):
        super().__init__(force_checkpoint_name)
        self.cache_folder = cache_folder
        self.force_checkpoint_name = force_checkpoint_name

    def read_checkpoint(self, data_container: DataContainer):
        """
        Read pickle files for data inputs and expected outputs checkpoint
        :return: tuple(data_inputs, expected_outputs
        """
        with open(self.get_checkpoint_file_path(data_container), 'rb') as file:
            checkpoint = pickle.load(file)

        return checkpoint

    def save_checkpoint(self, data_container: DataContainer):
        """
        Save pickle files for data inputs and expected output
        to create a checkpoint
        :param data_container: data inputs to be saved in a pickle file
        :return:
        """
        self.set_checkpoint_path(self.force_checkpoint_name)
        with open(self.get_checkpoint_file_path(data_container), 'wb') as file:
            pickle.dump(data_container, file)

    def set_checkpoint_path(self, path):
        """
        Set checkpoint path inside the cache folder (ex: cache_folder/pipeline_name/force_checkpoint_name/data_inputs.pickle)
        :param path: checkpoint path
        """
        if path is None:
            path = self.name

        self.checkpoint_path = os.path.join(self.cache_folder, path)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def should_resume(self, data_container: DataContainer) -> bool:
        return self.checkpoint_exists(data_container)

    def checkpoint_exists(self, data_container: DataContainer) -> bool:
        self.set_checkpoint_path(self.force_checkpoint_name)
        return os.path.exists(self.get_checkpoint_file_path(data_container))

    def get_checkpoint_file_path(self, data_container: DataContainer):
        return os.path.join(self.checkpoint_path, 'data_inputs.pickle')
