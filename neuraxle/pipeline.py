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
from typing import Any, Tuple

from neuraxle.base import BaseStep, TruncableSteps, NamedTupleList, ResumableStepMixin, DataContainer
from neuraxle.checkpoints import BaseCheckpointStep


class BasePipeline(TruncableSteps, ABC):
    def __init__(self, steps: NamedTupleList):
        TruncableSteps.__init__(self, steps_as_tuple=steps)

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


class Pipeline(BasePipeline, ResumableStepMixin):
    """
    Fits and transform steps after latest checkpoint
    """

    def __init__(self, steps: NamedTupleList):
        BasePipeline.__init__(self, steps=steps)

    def transform(self, data_inputs: Any):
        """
        After loading the last checkpoint, transform each pipeline steps

        :param data_inputs: the data input to transform
        :return: transformed data inputs
        """
        self.setup()

        current_ids = self.hasher.hash(
            current_ids=None,
            hyperparameters=self.hyperparams,
            data_inputs=data_inputs
        )
        data_container = DataContainer(current_ids=current_ids, data_inputs=data_inputs)
        data_container = self._transform_core(data_container)

        self.teardown()

        return data_container.data_inputs

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('Pipeline', Any):
        """
        After loading the last checkpoint, fit transform each pipeline steps

        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        self.setup()

        current_ids = self.hasher.hash(
            current_ids=None,
            hyperparameters=self.hyperparams,
            data_inputs=data_inputs
        )
        data_container = DataContainer(
            current_ids=current_ids,
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        )
        new_self, data_container = self._fit_transform_core(data_container)

        self.teardown()

        return new_self, data_container.data_inputs

    def fit(self, data_inputs, expected_outputs=None) -> 'Pipeline':
        """
        After loading the last checkpoint, fit each pipeline steps

        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        self.setup()

        current_ids = self.hasher.hash(
            current_ids=None,
            hyperparameters=self.hyperparams,
            data_inputs=data_inputs
        )
        data_container = DataContainer(
            current_ids=current_ids,
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        )
        new_self, _ = self._fit_transform_core(data_container)

        self.teardown()

        return new_self

    def inverse_transform_processed_outputs(self, processed_outputs) -> Any:
        """
        After transforming all data inputs, and obtaining a prediction, we can inverse transform the processed outputs

        :param processed_outputs: the forward transformed data input
        :return: backward transformed processed outputs
        """
        for step_name, step in list(reversed(self.items())):
            processed_outputs = step.transform(processed_outputs)
        return processed_outputs

    def handle_fit_transform(self, data_container: DataContainer) -> ('BaseStep', DataContainer):
        """
        Fit transform then rehash ids with hyperparams and transformed data inputs

        :param data_container: data container to fit transform
        :return: tuple(fitted pipeline, transformed data container)
        """
        new_self, data_container = self._fit_transform_core(data_container)

        ids = self.hasher.hash(data_container.current_ids, self.hyperparams, data_container.data_inputs)
        data_container.set_current_ids(ids)

        return new_self, data_container

    def handle_transform(self, data_container: DataContainer) -> DataContainer:
        """
        Transform then rehash ids with hyperparams and transformed data inputs

        :param data_container: data container to transform
        :return: tuple(fitted pipeline, transformed data container)
        """
        data_container = self._transform_core(data_container)

        ids = self.hasher.hash(data_container.current_ids, self.hyperparams, data_container.data_inputs)
        data_container.set_current_ids(ids)

        return data_container

    def _fit_transform_core(self, data_container) -> ('Pipeline', DataContainer):
        """
        After loading the last checkpoint, fit transform each pipeline steps

        :param data_container: the data container to fit transform on
        :return: tuple(pipeline, data_container)
        """
        steps_left_to_do, data_container = self._load_pipeline_checkpoint(data_container)

        new_steps_as_tuple: NamedTupleList = []

        for step_name, step in steps_left_to_do:
            step, data_container = step.handle_fit_transform(data_container)
            new_steps_as_tuple.append((step_name, step))

        self.steps_as_tuple = self.steps_as_tuple[:len(self.steps_as_tuple) - len(steps_left_to_do)] + \
                              new_steps_as_tuple

        return self, data_container

    def _transform_core(self, data_container: DataContainer) -> DataContainer:
        """
        After loading the last checkpoint, transform each pipeline steps

        :param data_container: the data container to transform
        :return: transformed data container
        """
        steps_left_to_do, data_container = self._load_pipeline_checkpoint(data_container)

        for step_name, step in steps_left_to_do:
            data_container = step.handle_transform(data_container)

        return data_container

    def _load_pipeline_checkpoint(self, data_container: DataContainer) -> Tuple[NamedTupleList, DataContainer]:
        """
        Find the steps left to do, and load the latest checkpoint step data container

        :param data_container: the data container to resume
        :return: tuple(steps left to do, last checkpoint data container)
        """
        new_starting_step_index = self._find_starting_step_index(data_container)

        step = self.steps_as_tuple[new_starting_step_index]
        if isinstance(step, BaseCheckpointStep):
            data_container = step.read_checkpoint(data_container)

        return self.steps_as_tuple[new_starting_step_index:], data_container

    def _find_starting_step_index(self, data_container: DataContainer) -> int:
        """
        Find the index of the latest step that can be resumed

        :param data_container: the data container to resume
        :return: int index latest resumable step
        """
        new_data_container = copy(data_container)
        index_latest_checkpoint = 0

        for index, (step_name, step) in enumerate(self.items()):
            if isinstance(step, ResumableStepMixin) and step.should_resume(new_data_container):
                data_container.set_current_ids(new_data_container.current_ids)
                index_latest_checkpoint = index

            current_ids = step.hasher.hash(
                current_ids=new_data_container.current_ids,
                hyperparameters=step.hyperparams,
                data_inputs=new_data_container.data_inputs
            )

            new_data_container.set_current_ids(current_ids)

        return index_latest_checkpoint

    def should_resume(self, data_container: DataContainer) -> bool:
        """
        Return True if the pipeline has a saved checkpoint that it can resume from

        :param data_container: the data container to resume
        :return: bool
        """
        for index, (step_name, step) in enumerate(reversed(self.items())):
            if isinstance(step, ResumableStepMixin) and step.should_resume(data_container):
                return True

        return False
