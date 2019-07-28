"""
Neuraxle Pickle Checkpoint Step
====================================
Checkpoint step that loads and saves pickles as checkpoints
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

from neuraxle.checkpoints.base_checkpoint_step import BaseCheckpointStep

DEFAULT_CACHE_FOLDER = os.path.join(os.getcwd(), 'cache')


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
