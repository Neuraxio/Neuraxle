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

from neuraxle.base import ResumableStepMixin, BaseStep

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

    def handle_transform(self, ids, data_inputs) -> Any:
        self.save_checkpoint(ids, data_inputs)
        return data_inputs

    def handle_fit_transform(self, ids, data_inputs, expected_outputs) -> ('BaseStep', Any):
        self.save_checkpoint(ids, data_inputs)
        return self, data_inputs

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
    def read_checkpoint(self, data_inputs: Any) -> Any:
        """
        Read checkpoint data to get the data inputs and expected output.
        :param data_inputs: data inputs to save
        :return: data_inputs_checkpoint
        """
        raise NotImplementedError()

    @abstractmethod
    def save_checkpoint(self, ids, data_inputs: Any):
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run
        :param ids: data inputs ids
        :param data_inputs: data inputs to save
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

    def read_checkpoint(self, data_inputs):
        """
        Read pickle files for data inputs and expected outputs checkpoint
        :return: tuple(data_inputs, expected_outputs
        """
        data_inputs_checkpoint_file_name = self.checkpoint_path
        with open(self.get_checkpoint_file_path(data_inputs_checkpoint_file_name), 'rb') as file:
            checkpoint = pickle.load(file)

        return checkpoint

    def save_checkpoint(self, ids, data_inputs):
        """
        Save pickle files for data inputs and expected output
        to create a checkpoint
        :param ids: data inputs ids
        :param data_inputs: data inputs to be saved in a pickle file
        :return:
        """
        # TODO: don't force the user to set the checkpoint name (use step name instead).
        self.set_checkpoint_path(self.force_checkpoint_name)
        with open(self.get_checkpoint_file_path(data_inputs), 'wb') as file:
            pickle.dump(data_inputs, file)

    def set_checkpoint_path(self, path):
        """
        Set checkpoint path inside the cache folder (ex: cache_folder/pipeline_name/force_checkpoint_name/data_inputs.pickle)
        :param path: checkpoint path
        """
        self.checkpoint_path = os.path.join(self.cache_folder, path)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def should_resume(self, data_inputs) -> bool:
        return self.checkpoint_exists(data_inputs)

    def checkpoint_exists(self, data_inputs) -> bool:
        self.set_checkpoint_path(self.force_checkpoint_name)
        return os.path.exists(self.get_checkpoint_file_path(data_inputs))

    def get_checkpoint_file_path(self, data_inputs):
        return os.path.join(self.checkpoint_path, 'data_inputs.pickle')
