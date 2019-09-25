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

    def setup(self, step_path: str, setup_arguments: dict):
        self.set_checkpoint_path(step_path)

    def handle_transform(self, data_container: DataContainer) -> DataContainer:
        return self.save_checkpoint(data_container)

    def handle_fit_transform(self, data_container: DataContainer) -> ('BaseStep', DataContainer):
        return self, self.save_checkpoint(data_container)

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
        :return: checkpoint data container
        """
        raise NotImplementedError()

    @abstractmethod
    def save_checkpoint(self, data_container: DataContainer) -> DataContainer:
        """
        Save checkpoint for data inputs and expected outputs so that it can
        be loaded by the checkpoint pipeline runner on the next pipeline run

        :param data_container: data inputs to save
        :return: saved data container
        """
        raise NotImplementedError()


class PickleCheckpointStep(BaseCheckpointStep):
    """
    Create pickles for a checkpoint step data inputs, and expected_outputs
    to eventually be able to load them using the checkpoint pipeline runner.
    """

    def __init__(self, cache_folder: str = DEFAULT_CACHE_FOLDER):
        super().__init__()
        self.cache_folder = cache_folder

    def read_checkpoint(self, data_container: DataContainer) -> DataContainer:
        """
        Read pickle files for data inputs and expected outputs checkpoint

        :return: tuple(data_inputs, expected_outputs
        """
        checkpoint_data_container = DataContainer(
            current_ids=[],
            data_inputs=[],
            expected_outputs=None
        )

        for current_id, data_input, expected_output in data_container:
            with open(self.get_checkpoint_file_path(current_id), 'wb') as file:
                (checkpoint_current_id, checkpoint_data_input, checkpoint_expected_output) = \
                    pickle.load(file)
                checkpoint_data_container.append(
                    current_id=checkpoint_current_id,
                    data_input=checkpoint_data_input,
                    expected_output=checkpoint_expected_output
                )

        return checkpoint_data_container

    def save_checkpoint(self, data_container: DataContainer) -> DataContainer:
        """
        Save pickle files for data inputs and expected output to create a checkpoint

        :param data_container: data to resume
        :return:
        """
        for current_id, data_input, expected_output in data_container:
            with open(self.get_checkpoint_file_path(current_id), 'wb') as file:
                pickle.dump(
                    (current_id, data_input, expected_output),
                    file
                )

        return data_container

    def set_checkpoint_path(self, path):
        """
        Set checkpoint path inside the cache folder (ex: cache_folder/pipeline/step_a/current_id.pickle)

        :param path: checkpoint path
        """
        self.checkpoint_path = os.path.join(self.cache_folder, path)
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path)

    def should_resume(self, data_container: DataContainer) -> bool:
        """
        Whether or not we should resume the pipeline (if the checkpoint exists)

        :param data_container: data to resume
        :return:
        """
        return self._checkpoint_exists(data_container)

    def _checkpoint_exists(self, data_container: DataContainer) -> bool:
        """
        Returns True if the checkpoints for each data input id exists
        :param data_container:
        :return:
        """
        for current_id in data_container.current_ids:
            if not os.path.exists(self.get_checkpoint_file_path(current_id)):
                return False

        return True

    def get_checkpoint_file_path(self, current_id) -> str:
        """
        Returns the checkpoint file path for a data input id

        :param current_id:
        :return:
        """
        return os.path.join(
            self.checkpoint_path,
            '{0}.pickle'.format(current_id)
        )
