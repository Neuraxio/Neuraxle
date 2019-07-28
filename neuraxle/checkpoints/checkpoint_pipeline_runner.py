"""
Neuraxle's Checkpoint Pipeline Runner
====================================
This is the runner that loads and save checkpoint steps.
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
from typing import Any

from neuraxle.base import BasePipelineRunner, NamedTupleList
from neuraxle.checkpoints.base_checkpoint_step import BaseCheckpointStep


class CheckpointPipelineRunner(BasePipelineRunner):
    """
    Checkpoint pipeline runner fits and transforms the steps after the latest checkpoint
    """

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('CheckpointPipelineRunner', Any):
        """
        After loading the last checkpoint, fit transform each pipeline steps
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        new_steps, data_inputs = self.fit_transform_steps(data_inputs, expected_outputs)
        processed_outputs = data_inputs

        return self, processed_outputs

    def fit(self, data_inputs, expected_outputs=None) -> 'CheckpointPipelineRunner':
        """
        After loading the last checkpoint, fit each pipeline steps
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        self.fit_transform_steps(data_inputs, expected_outputs)

        return self

    def fit_transform_steps(self, data_inputs, expected_outputs):
        """
        After loading the last checkpoint, fit transform each pipeline steps
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        self.create_checkpoint_names()
        (data_inputs, expected_outputs), index_starting_step = self.find_starting_step_and_data_inputs(data_inputs,
                                                                                                       expected_outputs)
        steps_left_to_do = self.steps_as_tuple[index_starting_step:]

        new_steps_as_tuple: NamedTupleList = []
        for step_name, step in steps_left_to_do:
            step, data_inputs = step.fit_transform(data_inputs, expected_outputs)

            if isinstance(step, BaseCheckpointStep):
                step.save_checkpoint(data_inputs, expected_outputs)

            new_steps_as_tuple.append((step_name, step))

        self.steps_as_tuple = new_steps_as_tuple

        return new_steps_as_tuple, data_inputs

    def transform(self, data_inputs):
        """
        After loading the last checkpoint, transform each pipeline steps
        :param data_inputs: the data input to fit on
        :return: transformed data inputs
        """
        self.create_checkpoint_names()
        (data_inputs, _), index_starting_step = self.find_starting_step_and_data_inputs(data_inputs)

        steps_left_to_do = self.steps_as_tuple[index_starting_step:]

        for step_name, step in steps_left_to_do:
            data_inputs = step.transform(data_inputs)

            if isinstance(step, BaseCheckpointStep):
                step.save_checkpoint(data_inputs)

        return data_inputs

    def find_starting_step_and_data_inputs(self, data_inputs, expected_outputs=None):
        """
        Find the starting step index, and its corresponding data inputs using the checkpoint steps
        :param expected_outputs: expected outputs to fit on
        :param data_inputs: the data input to fit on
        :return: tuple(tuple(data_inputs, expected_outputs), step_index) tuple for the starting step data inputs-outputs,
         and the starting step index
        """
        new_data_inputs = data_inputs
        new_expected_outputs = expected_outputs
        new_starting_step_index = 0

        for index, (step_name, step) in enumerate(reversed(self.steps_as_tuple)):
            if isinstance(step, BaseCheckpointStep):
                checkpoint_data_inputs, checkpoint_expected_data_inputs = step.load_checkpoint()

                if new_starting_step_index == 0 and \
                        checkpoint_data_inputs is not None:
                    new_data_inputs = checkpoint_data_inputs
                    new_expected_outputs = checkpoint_expected_data_inputs
                    new_starting_step_index = len(self.steps_as_tuple) - index - 1

        return (new_data_inputs, new_expected_outputs), new_starting_step_index

    def create_checkpoint_names(self):
        """
        Recursively create checkpoint step names based on their paths
        """
        new_steps_as_tuple: NamedTupleList = []

        for step_name, step in self.steps_as_tuple:
            if isinstance(step, BaseCheckpointStep):
                step.set_checkpoint_name(step_name)

            new_steps_as_tuple.append((step_name, step))

        self.steps_as_tuple = new_steps_as_tuple
