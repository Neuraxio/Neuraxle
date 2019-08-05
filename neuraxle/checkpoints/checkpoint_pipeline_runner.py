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

from neuraxle.base import BasePipelineRunner, NamedTupleList, TruncableSteps, MetaStepsMixin
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
        steps_as_tuple = self.steps_as_tuple

        new_steps_as_tuple = self.create_checkpoint_names(steps_as_tuple)
        self.steps_as_tuple = new_steps_as_tuple

        (data_inputs, expected_outputs), index_starting_step = \
            self.find_starting_step_and_data_inputs(
                steps_as_tuple=steps_as_tuple,
                data_inputs=data_inputs,
                expected_outputs=expected_outputs
            )

        steps_left_to_do = steps_as_tuple[index_starting_step:]

        new_steps_as_tuple, data_inputs = self.fit_transform_steps_left_to_do(data_inputs, expected_outputs,
                                                                              steps_left_to_do)

        self.steps_as_tuple = new_steps_as_tuple

        return new_steps_as_tuple, data_inputs

    def fit_transform_steps_left_to_do(self, data_inputs, expected_outputs, steps_left_to_do):
        """
        After loading the last checkpoint, fit transform each pipeline steps left to do
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :param steps_left_to_do: steps left to do
        :return: the pipeline itself
        """
        new_steps_as_tuple: NamedTupleList = []

        for step_name, step in steps_left_to_do:
            step, data_inputs = self.fit_transform_step(data_inputs, expected_outputs, step)
            new_steps_as_tuple.append((step_name, step))

        return new_steps_as_tuple, data_inputs

    def fit_transform_step(self, data_inputs, expected_outputs, step):
        """
        After loading the last checkpoint, fit transform step
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :param step: pipeline step
        :return: the pipeline itself
        """
        if isinstance(step, TruncableSteps) or isinstance(step, MetaStepsMixin):
            return self.fit_transform_steps_left_to_do_recursive(step,
                                                                 data_inputs,
                                                                 expected_outputs)
        else:
            return step.fit_transform(data_inputs, expected_outputs)

    def fit_transform_steps_left_to_do_recursive(self, step, data_inputs, expected_outputs):
        """
        After loading the last checkpoint, recursively fit transform a step that contains steps
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :param step: pipeline step
        :return: the pipeline itself
        """
        checkpoint, index = self.find_starting_step_and_data_inputs(
            steps_as_tuple=step.steps_as_tuple,
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        )

        checkpoint_data_input, checkpoint_expected_outputs = checkpoint
        if checkpoint_data_input is not None \
                and checkpoint_expected_outputs is not None:
            steps_left_to_do_recursive = step.steps_as_tuple[index:]
            return self.fit_transform_steps_left_to_do(checkpoint_data_input,
                                                       checkpoint_expected_outputs,
                                                       steps_left_to_do_recursive)

        return step.fit_transform(data_inputs, expected_outputs)

    def transform(self, data_inputs):
        """
        After loading the last checkpoint, transform each pipeline steps
        :param data_inputs: the data input to fit on
        :return: transformed data inputs
        """
        new_steps_as_tuple = self.create_checkpoint_names(self.steps_as_tuple)
        self.steps_as_tuple = new_steps_as_tuple

        (data_inputs, _), index_starting_step = self.find_starting_step_and_data_inputs(
            steps_as_tuple=self.steps_as_tuple,
            data_inputs=data_inputs
        )

        steps_left_to_do = self.steps_as_tuple[index_starting_step:]
        data_inputs = self.transform_steps_left_to_do(data_inputs, steps_left_to_do)

        return data_inputs

    def transform_steps_left_to_do(self, data_inputs, steps_left_to_do):
        """
        After loading the last checkpoint, transform each pipeline steps left to do
        :param data_inputs: the data input to transform on
        :param steps_left_to_do: steps left to do
        :return: the pipeline itself
        """
        for step_name, step in steps_left_to_do:
            if isinstance(step, TruncableSteps) or isinstance(step, MetaStepsMixin):
                data_inputs = self.transform_steps_left_to_do_recursive(step, data_inputs)
            else:
                data_inputs = step.transform(data_inputs)

        return data_inputs

    def transform_steps_left_to_do_recursive(self, step, data_inputs):
        """
        After loading the last checkpoint, recursively transform a step that contains steps
        :param data_inputs: the data input to transform on
        :param step: pipeline step
        :return: the pipeline itself
        """
        checkpoint, index = self.find_starting_step_and_data_inputs(
            steps_as_tuple=step.steps_as_tuple,
            data_inputs=data_inputs
        )

        checkpoint_data_input, checkpoint_expected_outputs = checkpoint
        if checkpoint_data_input is not None:
            steps_left_to_do_recursive = step.steps_as_tuple[index:]
            return self.transform_steps_left_to_do(checkpoint_data_input, steps_left_to_do_recursive)

        return step.transform(data_inputs)

    def find_starting_step_and_data_inputs(self, steps_as_tuple, data_inputs, expected_outputs=None):
        """
        Find the starting step index, and its corresponding data inputs using the checkpoint steps.
        If the checkpoint is inside another step, the starting step will be step that contains the checkpoint
        :param steps_as_tuple: steps
        :param expected_outputs: expected outputs to fit on
        :param data_inputs: the data input to fit on
        :return: tuple(tuple(data_inputs, expected_outputs), step_index) tuple for the starting step data inputs-outputs,
         and the starting step index
        """
        new_data_inputs = data_inputs
        new_expected_outputs = expected_outputs
        new_starting_step_index = 0
        checkpoint_new_data_inputs = None
        checkpoint_new_expected_outputs = None

        for index, (step_name, step) in enumerate(reversed(steps_as_tuple)):
            if isinstance(step, BaseCheckpointStep):
                checkpoint_new_data_inputs, checkpoint_new_expected_outputs = self.load_checkpoint(step)

            if isinstance(step, TruncableSteps) or isinstance(step, MetaStepsMixin):
                checkpoint_new_data_inputs, checkpoint_new_expected_outputs = self.load_checkpoint_recursive(
                    steps_as_tuple=step.steps_as_tuple,
                    data_inputs=data_inputs,
                    expected_outputs=expected_outputs
                )

            loaded_checkpoint_without_expected_outputs = expected_outputs is None and \
                                                         checkpoint_new_data_inputs is not None
            loaded_checkpoint_with_expected_outputs = expected_outputs is not None and \
                                                      checkpoint_new_data_inputs is not None and \
                                                      checkpoint_new_expected_outputs is not None

            if loaded_checkpoint_without_expected_outputs or loaded_checkpoint_with_expected_outputs:
                new_data_inputs = checkpoint_new_data_inputs
                new_starting_step_index = len(steps_as_tuple) - index - 1
                new_expected_outputs = checkpoint_new_expected_outputs if loaded_checkpoint_with_expected_outputs else new_expected_outputs
                return (new_data_inputs, new_expected_outputs), new_starting_step_index

        return (new_data_inputs, new_expected_outputs), new_starting_step_index

    def load_checkpoint(self, step):
        """
        Recursively find new starting step and data inputs for a step that contains other steps
        :param step: checkpoint step
        :return: tuple(tuple(data_inputs, expected_outputs), step_index) tuple for the starting step data inputs-outputs
        """
        checkpoint_data_inputs, checkpoint_expected_data_inputs = step.load_checkpoint()

        if checkpoint_data_inputs is not None:
            new_data_inputs = checkpoint_data_inputs
            new_expected_outputs = checkpoint_expected_data_inputs

            return new_data_inputs, new_expected_outputs

        return None, None

    def load_checkpoint_recursive(self, steps_as_tuple, data_inputs, expected_outputs=None):
        """
        Recursively find new starting step and data inputs for a step that contains other steps
        :param steps_as_tuple: steps of the current step being searched for a checkpoint
        :param expected_outputs: initial expected outputs of pipeline to load checkpoint from
        :param data_inputs: initial data inputs of pipeline to load checkpoint from
        :return: tuple(tuple(data_inputs, expected_outputs) tuple for the starting step data inputs-outputs
        """
        (new_data_inputs, new_expected_outputs), recursive_starting_step_index = \
            self.find_starting_step_and_data_inputs(steps_as_tuple, data_inputs, expected_outputs)

        if new_data_inputs is not None:
            return new_data_inputs, new_expected_outputs

        return None, None

    def create_checkpoint_names(self, steps_as_tuple):
        """
        Recursively create checkpoint step names based on their paths
        :param steps_as_tuple: steps of the current step being searched for a checkpoint
        """
        new_steps_as_tuple: NamedTupleList = []

        for step_name, step in steps_as_tuple:
            if isinstance(step, BaseCheckpointStep):
                step.set_checkpoint_name(step_name)

            if isinstance(step, TruncableSteps) or isinstance(step, MetaStepsMixin):
                new_steps_as_tuple_inner_step = self.create_checkpoint_names(step.steps_as_tuple)
                step.steps_as_tuple = new_steps_as_tuple_inner_step

            new_steps_as_tuple.append((step_name, step))

        return new_steps_as_tuple
