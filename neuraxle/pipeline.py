"""
Neuraxle's Pipeline Classes
====================================
This is the core of Neuraxle's pipelines. You can chain steps to call them one after an other.

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
from abc import ABC, abstractmethod
from copy import copy
from typing import Any

from neuraxle.base import BaseStep, TruncableSteps, NamedTupleList


class DataObject:
    def __init__(self, i, x):
        self.i = i
        self.x = x

        def __hash__(self):
            return hash((self.i, self.x))


class BasePipeline(TruncableSteps, ABC):

    def __init__(self, steps: NamedTupleList):
        BaseStep.__init__(self)
        TruncableSteps.__init__(self, steps)

    @abstractmethod
    def fit(self, data_inputs, expected_outputs=None) -> 'BasePipeline':
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data_inputs):
        raise NotImplementedError()

    @abstractmethod
    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BasePipeline', Any):
        raise NotImplementedError()

    def inverse_transform(self, processed_outputs):
        if self.transform != self.inverse_transform:
            raise BrokenPipeError("Don't call inverse_transform on a pipeline before having mutated it inversely or "
                                  "before having called the `.reverse()` or `reversed(.)` on it.")

        previous_steps_as_tuple = copy(self.steps_as_tuple)
        reversed_steps_as_tuple = list(reversed(self.steps_as_tuple))
        self.steps_as_tuple = reversed_steps_as_tuple
        processed_outputs = self.transform(processed_outputs)
        self.steps_as_tuple = previous_steps_as_tuple

        return processed_outputs


class ResumableStep(BaseStep):
    """
    A step that can be resumed, for example a checkpoint on disk.
    """

    """
    Load Pipeline Checkpoint
    :param data_inputs: initial data inputs
    :param expected_outputs: initial expected outputs
    :return: index, (data_inputs, expected_outputs)
    """
    @abstractmethod
    def load_checkpoint(self, data_inputs, expected_outputs=None):
        raise NotImplementedError()

    """
    Resume transform by loading the latest checkpoint
    :param data_inputs: initial data inputs
    :return: data_inputs
    """
    def resume_transform(self, data_inputs):
        index, checkpoint = self.load_checkpoint(data_inputs)
        checkpoint_data_input, _ = checkpoint

        if checkpoint_data_input is None:
            checkpoint_data_input = data_inputs

        return self.transform(checkpoint_data_input)

    """
    Resume fit transform by loading the latest checkpoint
    :param data_inputs: initial data inputs
    :return: data_inputs
    """
    def resume_fit_transform(self, data_inputs, expected_outputs):
        index, checkpoint = self.load_checkpoint(data_inputs, expected_outputs)
        checkpoint_data_input, checkpoint_expected_outputs = checkpoint

        if checkpoint_data_input is None or checkpoint_expected_outputs is None:
            checkpoint_data_input = data_inputs
            checkpoint_expected_outputs = expected_outputs

        return self.fit_transform(checkpoint_data_input, checkpoint_expected_outputs)


class Pipeline(BasePipeline, ResumableStep):
    """
    Fits and transform steps after latest checkpoint
    """

    def __init__(self, steps: NamedTupleList):
        super().__init__(steps)

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('Pipeline', Any):
        """
        After loading the last checkpoint, fit transform each pipeline steps
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        new_steps, data_inputs = self.fit_transform_steps(data_inputs, expected_outputs)
        processed_outputs = data_inputs

        return self, processed_outputs

    def fit(self, data_inputs, expected_outputs=None) -> 'Pipeline':
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
        (data_inputs, expected_outputs), index_starting_step = \
            self.find_starting_step_and_data_inputs(data_inputs, expected_outputs)

        steps_left_to_do = self.steps_as_tuple[index_starting_step:]

        new_steps_as_tuple: NamedTupleList = []
        for step_name, step in steps_left_to_do:
            if isinstance(step, ResumableStep):
                step, data_inputs = step.resume_fit_transform(data_inputs, expected_outputs)
            else:
                step, data_inputs = step.fit_transform(data_inputs, expected_outputs)
            new_steps_as_tuple.append((step_name, step))

        self.steps_as_tuple = new_steps_as_tuple

        return new_steps_as_tuple, data_inputs

    def transform(self, data_inputs):
        """
        After loading the last checkpoint, transform each pipeline steps
        :param data_inputs: the data input to fit on
        :return: transformed data inputs
        """
        (data_inputs, _), index_starting_step = self.find_starting_step_and_data_inputs(data_inputs)

        steps_left_to_do = self.steps_as_tuple[index_starting_step:]
        for step_name, step in steps_left_to_do:
            if isinstance(step, ResumableStep):
                data_inputs = step.resume_transform(data_inputs)
            else:
                data_inputs = step.transform(data_inputs)

        return data_inputs

    def find_starting_step_and_data_inputs(self, data_inputs, expected_outputs=None):
        """
        Find the starting step index, and its corresponding data inputs using the checkpoint steps.
        If the checkpoint is inside another step, the starting step will be step that contains the checkpoint
        :param expected_outputs: expected outputs to fit on
        :param data_inputs: the data input to fit on
        :return: tuple(tuple(data_inputs, expected_outputs), step_index) tuple for the starting step data inputs-outputs,
         and the starting step index
        """
        new_data_inputs = data_inputs
        new_expected_outputs = expected_outputs
        new_starting_step_index = 0

        for index, (step_name, step) in enumerate(reversed(self.steps_as_tuple)):
            if isinstance(step, ResumableStep):
                _, checkpoint = step.load_checkpoint(data_inputs, expected_outputs)
                (checkpoint_data_inputs, checkpoint_expected_outputs) = checkpoint

                if checkpoint_data_inputs is not None:
                    new_starting_step_index = len(self.steps_as_tuple) - index - 1
                    return (checkpoint_data_inputs, checkpoint_expected_outputs), new_starting_step_index

        return (new_data_inputs, new_expected_outputs), new_starting_step_index

    def load_checkpoint(self, data_inputs, expected_outputs=None):
        """
        Load Pipeline Checkpoint
        :param data_inputs: initial data inputs of pipeline to load checkpoint from
        :param expected_outputs: initial expected outputs of pipeline to load checkpoint from
        :return: tuple(tuple(data_inputs, expected_outputs), index)
        """
        (new_data_inputs, new_expected_outputs), index = \
            self.find_starting_step_and_data_inputs(data_inputs, expected_outputs)

        if expected_outputs is not None and new_expected_outputs is None:
            return (None, None), 0

        return (new_data_inputs, new_expected_outputs), 0
