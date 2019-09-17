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
import inspect
import glob
import os
from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Tuple, Optional

from joblib import load, dump

from neuraxle.base import BaseStep, TruncableSteps, NamedTupleList, ResumableStepMixin, DataContainer
from neuraxle.checkpoints import BaseCheckpointStep

DEFAULT_CACHE_FOLDER = 'cache'


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


class PipelineSaver(ABC):
    @abstractmethod
    def save(self, pipeline: 'Pipeline', data_container: DataContainer) -> 'Pipeline':
        """
        Persist pipeline for current data container

        :param pipeline:
        :param data_container:
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def load(self, pipeline: 'Pipeline', data_container: DataContainer) -> 'Pipeline':
        """
        Load pipeline for current data container

        :param pipeline:
        :param data_container:
        :return: pipeline
        """
        raise NotImplementedError()


class JoblibPipelineSaver(PipelineSaver):
    """
    Pipeline Repository to persist and load pipeline based on a data container
    """

    def __init__(self, cache_folder, pipeline_cache_list_file_name='pipeline_cache_list.txt'):
        self.pipeline_cache_list_file_name = pipeline_cache_list_file_name
        self.cache_folder = cache_folder

    def save(self, pipeline: 'Pipeline', data_container: DataContainer) -> 'Pipeline':
        """
        Persist pipeline for current data container

        :param pipeline:
        :param data_container:
        :return:
        """
        pipeline_cache_folder = os.path.join(self.cache_folder, pipeline.name)

        if not os.path.exists(pipeline_cache_folder):
            os.makedirs(pipeline_cache_folder)

        with open(os.path.join(pipeline_cache_folder, self.pipeline_cache_list_file_name), mode='a') as file_:
            next_cached_pipeline_path = self._create_next_cached_pipeline_path(pipeline_cache_folder)

            file_.write(str(data_container))
            file_.write('\n')
            file_.write(next_cached_pipeline_path)
            file_.write('\n')

            dump(pipeline, next_cached_pipeline_path)

        return pipeline

    def load(self, pipeline: 'Pipeline', data_container: DataContainer) -> Optional['Pipeline']:
        """
        Load pipeline for current data container

        :param pipeline:
        :param data_container:
        :return: pipeline
        """
        pipeline_cache_folder = os.path.join(self.cache_folder, pipeline.name)
        pipeline_cache_list_file_name_path = os.path.join(pipeline_cache_folder, self.pipeline_cache_list_file_name)
        if not os.path.exists(pipeline_cache_list_file_name_path):
            return None

        with open(pipeline_cache_list_file_name_path, mode='r') as file:
            found_cached_pipeline = False
            for line in file.readlines():
                if found_cached_pipeline:
                    return load(line.strip())

                if str(data_container) == line.strip():
                    found_cached_pipeline = True

        return None

    def _create_next_cached_pipeline_path(self, pipeline_cache_folder) -> str:
        """
        Create next cached pipeline path by incrementing a suffix

        :param pipeline_cache_folder:
        :return: path string
        """
        cached_pipeline_paths = [path for path in glob.glob(os.path.join(pipeline_cache_folder, '*'))]
        pipeline_file_name_index = 0

        while self._create_cached_pipeline_path(pipeline_cache_folder,
                                                pipeline_file_name_index) in cached_pipeline_paths:
            pipeline_file_name_index += 1

        return self._create_cached_pipeline_path(pipeline_cache_folder, pipeline_file_name_index)

    def _create_cached_pipeline_path(self, pipeline_cache_folder, pipeline_file_name_index):
        return os.path.join(pipeline_cache_folder,
                            '{0}.joblib'.format(str(pipeline_file_name_index)))


class Pipeline(BasePipeline):
    """
    Fits and transform steps
    """

    def __init__(self, steps: NamedTupleList):
        BasePipeline.__init__(self, steps=steps)

    def save(self, data_container: DataContainer):
        """
        Save the fitted parent pipeline with the passed data container

        :param data_container: data container to save pipeline with
        :return:
        """
        if self.parent_step is None:
            pass  # nothing to do here, we cannot save a pipeline that is not resumable
        else:
            self.parent_step.save(data_container)

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
        steps_left_to_do, data_container = self._load_checkpoint(data_container)

        new_steps_as_tuple: NamedTupleList = []

        for step_name, step in steps_left_to_do:
            step.set_parent(self)
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
        steps_left_to_do, data_container = self._load_checkpoint(data_container)

        for step_name, step in steps_left_to_do:
            step.set_parent(self)
            data_container = step.handle_transform(data_container)

        return data_container

    def _load_checkpoint(self, data_container: DataContainer) -> Tuple[NamedTupleList, DataContainer]:
        """
        Try loading a pipeline cache with the passed data container.
        If pipeline cache loading succeeds, find steps left to do,
        and load the latest data container.

        :param data_container: the data container to resume
        :return: tuple(steps left to do, last checkpoint data container)
        """
        return self.steps_as_tuple, data_container


class ResumablePipeline(Pipeline, ResumableStepMixin):
    """
    Fits and transform steps after latest checkpoint
    """

    def __init__(self, steps: NamedTupleList, pipeline_saver: PipelineSaver = None):
        Pipeline.__init__(self, steps=steps)

        if pipeline_saver is None:
            self.pipeline_saver = JoblibPipelineSaver(DEFAULT_CACHE_FOLDER)
        else:
            self.pipeline_saver = pipeline_saver

    def save(self, data_container: DataContainer):
        """
        Save the fitted parent pipeline with the passed data container

        :param data_container: data container to save pipeline with
        :return:
        """
        if self.parent_step is None:
            self.pipeline_saver.save(self, data_container)
        else:
            self.parent_step.save(data_container)

    def _load_checkpoint(self, data_container: DataContainer) -> Tuple[NamedTupleList, DataContainer]:
        """
        Try loading a pipeline cache with the passed data container.
        If pipeline cache loading succeeds, find steps left to do,
        and load the latest data container.

        :param data_container: the data container to resume
        :return: tuple(steps left to do, last checkpoint data container)
        """
        new_starting_step_index, starting_step_data_container = \
            self._get_starting_step_index_and_data_container(data_container)

        if self.parent_step is None:
            loaded_pipeline = self._try_loading_pipeline_checkpoint(
                starting_step_data_container,
                new_starting_step_index
            )

            if not loaded_pipeline:
                return self.steps_as_tuple, data_container

        step = self.steps_as_tuple[new_starting_step_index]
        if isinstance(step, BaseCheckpointStep):
            starting_step_data_container = step.read_checkpoint(starting_step_data_container)

        return self.steps_as_tuple[new_starting_step_index:], starting_step_data_container

    def _try_loading_pipeline_checkpoint(
            self,
            starting_step_data_container: DataContainer,
            new_starting_step_index
    ) -> bool:
        """
        Load persisted pipeline steps before checkpoint
        if the steps before the latest checkpoint have not changed

        :param starting_step_data_container: loaded cached pipeline
        :param new_starting_step_index:
        :return: true if loading succeeded
        """
        cached_pipeline = self.pipeline_saver.load(self, starting_step_data_container)
        if cached_pipeline is None:
            return False

        if self.compare_other_truncable_steps_before_index(cached_pipeline, new_starting_step_index):
            self.load_other_truncable_steps_before_index(cached_pipeline, new_starting_step_index,)
            return True

        return False

    def _get_starting_step_index_and_data_container(self, data_container: DataContainer) -> Tuple[int, DataContainer]:
        """
        Find the index of the latest step that can be resumed

        :param data_container: the data container to resume
        :return: index of the latest resumable step, data container at starting step
        """
        starting_step_data_container = copy(data_container)
        current_data_container = copy(data_container)
        index_latest_checkpoint = 0

        for index, (step_name, step) in enumerate(self.items()):
            if isinstance(step, ResumableStepMixin) and step.should_resume(current_data_container):
                index_latest_checkpoint = index
                starting_step_data_container = copy(current_data_container)

            current_ids = step.hasher.hash(
                current_ids=current_data_container.current_ids,
                hyperparameters=step.hyperparams,
                data_inputs=current_data_container.data_inputs
            )

            current_data_container.set_current_ids(current_ids)

        return index_latest_checkpoint, starting_step_data_container

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
