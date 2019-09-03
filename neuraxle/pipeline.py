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
from typing import Any, Tuple

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

    @abstractmethod
    def inverse_transform_processed_outputs(self, data_inputs) -> Any:
        raise NotImplementedError()

    def inverse_transform(self, processed_outputs):
        if self.transform != self.inverse_transform:
            raise BrokenPipeError("Don't call inverse_transform on a pipeline before having mutated it inversely or "
                                  "before having called the `.reverse()` or `reversed(.)` on it.")

        return self.inverse_transform_processed_outputs(processed_outputs)


class ResumableStep(BaseStep):
    """
    A step that can be resumed, for example a checkpoint on disk.
    """

    """
    Load Pipeline Checkpoint
    :param data_inputs: initial data inputs
    :return: steps_left_to_do, data_inputs
    """
    @abstractmethod
    def load_checkpoint(self, data_inputs) -> Tuple[list, Any]:
        raise NotImplementedError()

    @abstractmethod
    def should_resume(self, data_inputs) -> bool:
        raise NotImplementedError()


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
        new_self, data_inputs = self.fit_transform_steps(data_inputs, expected_outputs)
        processed_outputs = data_inputs

        return new_self, processed_outputs

    def fit(self, data_inputs, expected_outputs=None) -> 'Pipeline':
        """
        After loading the last checkpoint, fit each pipeline steps
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        new_self, _ = self.fit_transform_steps(data_inputs, expected_outputs)

        return new_self

    def fit_transform_steps(self, data_inputs, expected_outputs):
        """
        After loading the last checkpoint, fit transform each pipeline steps
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        steps_left_to_do, data_inputs = self.load_checkpoint(data_inputs)

        new_steps_as_tuple: NamedTupleList = []
        for step_name, step in steps_left_to_do:
            step, data_inputs = step.fit_transform(data_inputs, expected_outputs)
            new_steps_as_tuple.append((step_name, step))

        self.steps_as_tuple = self.steps_as_tuple[:len(self.steps_as_tuple) - len(steps_left_to_do)] + \
                              new_steps_as_tuple

        return self, data_inputs

    def transform(self, data_inputs):
        """
        After loading the last checkpoint, transform each pipeline steps
        :param data_inputs: the data input to fit on
        :return: transformed data inputs
        """
        steps_left_to_do, data_inputs = self.load_checkpoint(data_inputs)
        for step_name, step in steps_left_to_do:
            data_inputs = step.transform(data_inputs)

        return data_inputs

    def inverse_transform_processed_outputs(self, processed_outputs) -> Any:
        """
        After transforming all data inputs, and obtaining a prediction, we can inverse transform the processed outputs
        :param processed_outputs: the forward transformed data input
        :return: backward transformed processed outputs
        """
        for step_name, step in list(reversed(self.steps_as_tuple)):
            processed_outputs = step.transform(processed_outputs)
        return processed_outputs

    def load_checkpoint(self, data_inputs, expected_outputs=None):
        """
        Find the starting step index, and its corresponding data inputs using the checkpoint steps.
        If the checkpoint is inside another step, the starting step will be step that contains the checkpoint
        :param expected_outputs: expected outputs to fit on
        :param data_inputs: the data input to fit on
        :return: index, (data_inputs, expected_outputs) tuple for the starting step data inputs-outputs
         and the starting step index
        """
        new_data_inputs = data_inputs
        new_starting_step_index = 0
        found_checkpoint = False

        for index, (step_name, step) in enumerate(reversed(self.steps_as_tuple)):
            if not found_checkpoint and isinstance(step, ResumableStep) and step.should_resume(data_inputs):
                _, checkpoint = step.load_checkpoint(data_inputs)
                new_starting_step_index = len(self.steps_as_tuple) - index - 1
                new_data_inputs = checkpoint
                found_checkpoint = True

        return self.steps_as_tuple[new_starting_step_index:], new_data_inputs

    def should_resume(self, data_inputs) -> bool:
        """
        Return True if the pipeline has a step that can be resumed (another pipeline or a checkpoint step).
        :param data_inputs: the data input to fit on
        :return: boolean
        """
        should_resume = False
        for index, (step_name, step) in enumerate(reversed(self.steps_as_tuple)):
            if isinstance(step, ResumableStep) and step.should_resume(data_inputs):
                should_resume = True

        return should_resume
