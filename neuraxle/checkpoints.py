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
from abc import abstractmethod, ABC
from neuraxle.base import BaseStep
import os
import pickle

DEFAULT_CACHE_FOLDER = os.path.join(os.getcwd(), 'cache')


class BaseCheckpointStep(BaseStep, ABC):
    """
    Base class for a checkpoint step that can persists the received data inputs, and expected_outputs
    to eventually be able to load them using the checkpoint pipeline runner.
    """

    def __init__(self, force_checkpoint_name=None):
        super().__init__()
        self.force_checkpoint_name = force_checkpoint_name
        self.checkpoint_name = force_checkpoint_name

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseCheckpointStep':
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run
        :param data_inputs: data inputs to save
        :return: self
        """
        self.save_checkpoint(data_inputs, expected_outputs)

        return self

    def transform(self, data_inputs):
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run
        :param data_inputs: data inputs to save
        :return: data_inputs
        """
        self.save_checkpoint(data_inputs)

        return data_inputs

    def set_checkpoint_name(self, checkpoint_name) -> 'BaseCheckpointStep':
        """
        Set checkpoint name.
        :param checkpoint_name: str for checkpoint name
        :return: tuple(data_inputs, expected_outputs)
        """
        if self.force_checkpoint_name is None:
            self.checkpoint_name = checkpoint_name
        else:
            self.checkpoint_name = self.force_checkpoint_name

        return self

    @abstractmethod
    def load_checkpoint(self):
        """
        Load checkpoint to get the data inputs and expected output.
        Note: This method is called by the checkpoint pipeline runner
        to start at the latest checkpoint
        :return: tuple(data_inputs, expected_outputs)
        """
        raise NotImplementedError()

    @abstractmethod
    def save_checkpoint(self, data_inputs, expected_output=None):
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run
        :param data_inputs: data inputs to save
        :param expected_output: expected output to save
        :return:
        """
        raise NotImplementedError()


class PickleCheckpointStep(BaseCheckpointStep):
    """
    Create pickles for a checkpoint step data inputs, and expected_outputs
    to eventually be able to load them using the checkpoint pipeline runner.
    """

    def __init__(self, force_checkpoint_name=None, checkpoint_folder=DEFAULT_CACHE_FOLDER):
        super().__init__(force_checkpoint_name)
        self.checkpoint_folder = checkpoint_folder

    def load_checkpoint(self):
        """
        Read pickle files for data inputs and expected outputs checkpoint
        :return: tuple(data_inputs, expected_outputs
        """
        data_inputs = None
        expected_outputs = None

        data_inputs_checkpoint_file_name = self.get_data_inputs_checkpoint_file_name()
        if os.path.exists(data_inputs_checkpoint_file_name):
            with open(data_inputs_checkpoint_file_name, 'rb') as file:
                data_inputs = pickle.load(file)

        expected_outputs_checkpoint_file_name = self.get_expected_ouputs_file_name()
        if os.path.exists(expected_outputs_checkpoint_file_name):
            with open(expected_outputs_checkpoint_file_name, 'rb') as file:
                expected_outputs = pickle.load(file)

        return data_inputs, expected_outputs

    def save_checkpoint(self, data_inputs, expected_outputs=None):
        """
        Save pickle files for data inputs and expected output
        to create a checkpoint
        :param data_inputs: data inputs to be saved in a pickle file
        :param expected_outputs: expected outputs to be saved in a pickle file
        :return:
        """
        with open(self.get_data_inputs_checkpoint_file_name(), 'wb') as file:
            pickle.dump(data_inputs, file)

        with open(self.get_expected_ouputs_file_name(), 'wb') as file:
            pickle.dump(data_inputs, file)

    def get_expected_ouputs_file_name(self):
        return '{0}/{1}_expected_outputs'.format(self.checkpoint_folder, self.checkpoint_name)

    def get_data_inputs_checkpoint_file_name(self):
        return '{0}/{1}_data_inputs'.format(self.checkpoint_folder, self.checkpoint_name)
