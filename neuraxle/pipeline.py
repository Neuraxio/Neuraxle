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
import hashlib
import os
from abc import ABC, abstractmethod
from copy import copy
from typing import Any, Tuple, List, Iterable

from conv import convolved_1d
from joblib import load, dump

from neuraxle.base import BaseStep, TruncableSteps, NamedTupleList, ResumableStepMixin, DataContainer, NonFittableMixin, \
    NonTransformableMixin
from neuraxle.checkpoints import BaseCheckpointStep

BARRIER_STEP_NAME = 'Barrier'

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
    def can_load(self, pipeline: 'Pipeline', data_container: DataContainer) -> bool:
        """
        Returns True if the pipeline can be loaded with the passed data container

        :param pipeline:
        :param data_container:
        :return: pipeline
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

        next_cached_pipeline_path = self._create_cached_pipeline_path(pipeline_cache_folder, data_container)
        dump(pipeline, next_cached_pipeline_path)

        return pipeline

    def can_load(self, pipeline: 'Pipeline', data_container: DataContainer) -> bool:
        """
        Returns True if the pipeline can be loaded with the passed data container

        :param pipeline:
        :param data_container:
        :return: pipeline
        """
        pipeline_cache_folder = os.path.join(self.cache_folder, pipeline.name)

        return os.path.exists(
            self._create_cached_pipeline_path(
                pipeline_cache_folder,
                data_container
            )
        )

    def load(self, pipeline: 'Pipeline', data_container: DataContainer) -> 'Pipeline':
        """
        Load pipeline for current data container.

        :param pipeline:
        :param data_container:
        :return: pipeline
        """
        pipeline_cache_folder = os.path.join(self.cache_folder, pipeline.name)

        return load(
            self._create_cached_pipeline_path(
                pipeline_cache_folder,
                data_container
            )
        )

    def _create_cached_pipeline_path(self, pipeline_cache_folder, data_container: 'DataContainer') -> str:
        """
        Create cached pipeline path with data container, and pipeline cache folder.

        :type data_container: DataContainer
        :param pipeline_cache_folder: str

        :return: path string
        """
        all_current_ids_hash = self._hash_data_container(data_container)
        return os.path.join(pipeline_cache_folder, '{0}.joblib'.format(str(all_current_ids_hash)))

    def _hash_data_container(self, data_container):
        """
        Hash data container current ids with md5.

        :param data_container: data container
        :type data_container: DataContainer

        :return: str hexdigest of all of the current ids hashed together.
        """
        all_current_ids_hash = None
        for current_id, *_ in data_container:
            m = hashlib.md5()
            m.update(str.encode(current_id))
            if all_current_ids_hash is not None:
                m.update(str.encode(all_current_ids_hash))
            all_current_ids_hash = m.hexdigest()
        return all_current_ids_hash


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
        self.setup()  # TODO: perhaps, remove this to pass path in context

        current_ids = self.hash(
            current_ids=None,
            hyperparameters=self.hyperparams,
            data_inputs=data_inputs
        )
        data_container = DataContainer(current_ids=current_ids, data_inputs=data_inputs)
        data_container = self._transform_core(data_container)

        return data_container.data_inputs

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('Pipeline', Any):
        """
        After loading the last checkpoint, fit transform each pipeline steps

        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        self.setup()  # TODO: perhaps, remove this to pass path in context

        current_ids = self.hash(
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

        return new_self, data_container.data_inputs

    def fit(self, data_inputs, expected_outputs=None) -> 'Pipeline':
        """
        After loading the last checkpoint, fit each pipeline steps

        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        self.setup()  # TODO: perhaps, remove this to pass path in context

        current_ids = self.hash(
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

        ids = self.hash(data_container.current_ids, self.hyperparams, data_container.data_inputs)
        data_container.set_current_ids(ids)

        return new_self, data_container

    def handle_transform(self, data_container: DataContainer) -> DataContainer:
        """
        Transform then rehash ids with hyperparams and transformed data inputs

        :param data_container: data container to transform
        :return: tuple(fitted pipeline, transformed data container)
        """
        data_container = self._transform_core(data_container)

        ids = self.hash(data_container.current_ids, self.hyperparams, data_container.data_inputs)
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
            self._get_starting_step_info(data_container)

        if self.parent_step is None:
            if not self.pipeline_saver.can_load(self, starting_step_data_container):
                return self.steps_as_tuple, data_container

            saved_pipeline = self.pipeline_saver.load(self, starting_step_data_container)

            if not self._can_load_saved_pipeline(
                    saved_pipeline=saved_pipeline,
                    starting_step_data_container=starting_step_data_container,
                    new_starting_step_index=new_starting_step_index
            ):
                return self.steps_as_tuple, data_container

            self._load_saved_pipeline_steps_before_index(
                saved_pipeline=saved_pipeline,
                index=new_starting_step_index
            )

        step = self[new_starting_step_index]
        if isinstance(step, BaseCheckpointStep):
            starting_step_data_container = step.read_checkpoint(starting_step_data_container)

        return self[new_starting_step_index:], starting_step_data_container

    def _can_load_saved_pipeline(
            self,
            saved_pipeline: Pipeline,
            starting_step_data_container: DataContainer,
            new_starting_step_index
    ) -> bool:
        """
        Returns True if the saved pipeline steps before passed starting step index
        are the same as current pipeline steps before starting step index.

        :param saved_pipeline: loaded saved pipeline
        :param starting_step_data_container: loaded cached pipeline
        :param new_starting_step_index:

        :return bool
        """
        if not self.pipeline_saver.can_load(self, starting_step_data_container):
            return False

        if self.are_steps_before_index_the_same(saved_pipeline, new_starting_step_index):
            return True

        return False

    def _get_starting_step_info(self, data_container: DataContainer) -> Tuple[int, DataContainer]:
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

            current_ids = step.hash(
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


"""
Idea for checkpoints : 

    The streaming pipeline algorithm could go find the optional checkpoint step for each sub pipeline.
    In the future, a ResumableStreamingPipeline that extends this class should exist to support checkpoints.
    
    The Barrier should ideally join the data so that the ram does not blow up (iterable, lazy loading, cache ??)
    Maybe we can implement a LazyLoadingDataContainer/CachedDataContainer class or something that could be returned.
    
    pipeline = Pipeline([
        MiniBatchSequentialPipeline([

            A(),
            B(),
            Barrier(joiner=Joiner()),
            [Checkpoint()]

            C(),
            D(),
            Barrier(joiner=Joiner()),
            [Checkpoint()]

        ]),
        Model()
    ])
    
    pipeline = Pipeline([
        MiniBatchSequentialPipeline([
            ParallelWrapper([
                NonFittableA(),
                NonFittableB(),
            ])
            Barrier(joiner=Joiner()),
            [Checkpoint()]

            C(),
            D(),
            Barrier(joiner=Joiner()),
            [Checkpoint()]

        ]),
        Model()
    ])
"""


class Barrier(NonFittableMixin, NonTransformableMixin, BaseStep):
    pass


class MiniBatchSequentialPipeline(NonFittableMixin, Pipeline):
    """
    Streaming Pipeline class to create a pipeline for streaming, and batch processing.
    """

    def __init__(self, steps: NamedTupleList, batch_size):
        Pipeline.__init__(self, steps)
        self.batch_size = batch_size

    def transform(self, data_inputs: Any):
        """
        :param data_inputs: the data input to transform
        :return: transformed data inputs
        """
        self.setup()

        current_ids = self.hash(
            current_ids=None,
            hyperparameters=self.hyperparams,
            data_inputs=data_inputs
        )
        data_container = DataContainer(current_ids=current_ids, data_inputs=data_inputs)
        data_container = self.handle_transform(data_container)

        return data_container.data_inputs

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('Pipeline', Any):
        """
        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        self.setup()

        current_ids = self.hash(
            current_ids=None,
            hyperparameters=self.hyperparams,
            data_inputs=data_inputs
        )
        data_container = DataContainer(
            current_ids=current_ids,
            data_inputs=data_inputs,
            expected_outputs=expected_outputs
        )
        new_self, data_container = self.handle_fit_transform(data_container)

        return new_self, data_container.data_inputs

    def handle_transform(self, data_container: DataContainer) -> DataContainer:
        """
        Transform all sub pipelines splitted by the Barrier steps.

        :param data_container: data container to transform.
        :return: data container
        """
        sub_pipelines = self._create_sub_pipelines()

        for sub_pipeline in sub_pipelines:
            data_container = self._handle_transform_sub_pipeline(sub_pipeline, data_container)

        return data_container

    def handle_fit_transform(self, data_container: DataContainer) -> \
            Tuple['MiniBatchSequentialPipeline', DataContainer]:
        """
        Transform all sub pipelines splitted by the Barrier steps.

        :param data_container: data container to transform.
        :return: data container
        """
        sub_pipelines = self._create_sub_pipelines()

        for sub_pipeline in sub_pipelines:
            new_self, data_container = self._handle_fit_transform_sub_pipeline(sub_pipeline, data_container)

        return self, data_container

    def _handle_transform_sub_pipeline(self, sub_pipeline, data_container) -> DataContainer:
        """
        Transform sub pipeline using join transform.

        :param sub_pipeline: sub pipeline to be used to transform data container
        :param data_container: data container to transform
        :return:
        """
        data_inputs = sub_pipeline.join_transform(data_container.data_inputs)
        data_container.set_data_inputs(data_inputs)

        current_ids = self.hash(data_container.current_ids, self.hyperparams, data_inputs)
        data_container.set_current_ids(current_ids)

        return data_container

    def _handle_fit_transform_sub_pipeline(self, sub_pipeline, data_container) -> \
            Tuple['MiniBatchSequentialPipeline', DataContainer]:
        """
        Fit Transform sub pipeline using join fit transform.

        :param sub_pipeline: sub pipeline to be used to transform data container
        :param data_container: data container to fit transform
        :return: fitted self, transformed data container
        """
        _, data_inputs = sub_pipeline.join_fit_transform(
            data_inputs=data_container.data_inputs,
            expected_outputs=data_container.expected_outputs
        )
        data_container.set_data_inputs(data_inputs)

        current_ids = self.hash(data_container.current_ids, self.hyperparams, data_inputs)
        data_container.set_current_ids(current_ids)

        return self, data_container

    def _create_sub_pipelines(self) -> List['MiniBatchSequentialPipeline']:
        """
        Create sub pipelines by splitting the steps by the join type name.

        :return: list of sub pipelines
        """
        sub_pipelines: List[MiniBatchSequentialPipeline] = self.split(BARRIER_STEP_NAME)
        for sub_pipeline in sub_pipelines:
            if not sub_pipeline.ends_with_type_name(BARRIER_STEP_NAME):
                raise Exception(
                    'At least one Barrier step needs to be at the end of a streaming pipeline. '.format(
                        self.join_type_name)
                )

        return sub_pipelines

    def join_transform(self, data_inputs: Iterable) -> Iterable:
        """
        Concatenate the transform output of each batch of self.batch_size together.

        :param data_inputs:
        :return:
        """
        outputs = []
        for batch in convolved_1d(
                stride=self.batch_size,
                iterable=data_inputs,
                kernel_size=self.batch_size
        ):
            batch_outputs = super().transform(batch)
            outputs.extend(batch_outputs)  # TODO: use a joiner here

        return outputs

    def join_fit_transform(self, data_inputs: Iterable, expected_outputs: Iterable = None) -> \
            Tuple['MiniBatchSequentialPipeline', Iterable]:
        """
        Concatenate the fit transform output of each batch of self.batch_size together.

        :param data_inputs: data inputs to fit transform on
        :param expected_outputs: expected outputs to fit
        :return: fitted self, transformed data inputs
        """
        outputs = []
        for batch in convolved_1d(
                stride=self.batch_size,
                iterable=zip(data_inputs, expected_outputs),
                kernel_size=self.batch_size
        ):
            di_eo_list = list(zip(*batch))
            _, batch_outputs = super().fit_transform(list(di_eo_list[0]), list(di_eo_list[1]))
            outputs.extend(batch_outputs)  # TODO: use a joiner here

        return self, outputs
