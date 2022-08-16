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

..
    Thanks to Umaneo Technologies Inc. for their contributions to this Machine Learning
    project, visit https://www.umaneo.com/ for more information on Umaneo Technologies Inc.

"""
import warnings
from abc import ABC, abstractmethod
from typing import Any, List, Tuple, Union

from neuraxle.base import BaseStep
from neuraxle.base import ExecutionContext as CX
from neuraxle.base import (ExecutionMode, ForceHandleMixin, Identity, NamedStepsList, TruncableSteps,
                           _CustomHandlerMethods, _TruncableServiceWithBodyMixin)
from neuraxle.data_container import DataContainer as DACT
from neuraxle.data_container import ListDataContainer, StripAbsentValues, ZipDataContainer
from neuraxle.logging.warnings import warn_deprecated_arg


class BasePipeline(TruncableSteps, ABC):
    """
    Pipeline is a list of steps. This base class is the base class for all pipelines.

    .. seealso::
        :class:`~neuraxle.pipeline.Pipeline`,
        :class:`~neuraxle.pipeline.MiniBatchSequentialPipeline`,
        :class:`~neuraxle.distributed.streaming.SequentialQueuedPipeline`
    """

    def __init__(self, steps: NamedStepsList):
        TruncableSteps.__init__(self, steps_as_tuple=steps)

    @abstractmethod
    def fit(self, data_inputs, expected_outputs=None) -> 'BasePipeline':
        raise NotImplementedError()

    @abstractmethod
    def transform(self, data_inputs):
        raise NotImplementedError()

    @abstractmethod
    def fit_transform(self, data_inputs, expected_outputs=None) -> Tuple['BasePipeline', Any]:
        raise NotImplementedError()


class Pipeline(BasePipeline):
    """
    Handle methods are used to handle the fit, transform, fit_transform and inverse_transform methods
    using a pipe and filter design pattern to handle the steps sequentially. Each step is called
    with the previous step's output as input. The last step's output is returned as final output.

    Think like the termina pipe commands chained together with the character '|', but where each
    command is a step, and where each step can acquire some state from fitting the transformed
    output of the previous step. This way, the steps can be chained together and the state of each
    step can be saved and reused in the end.

    When a pipeline fit, a new pipeline is returned containing the fitted steps within itself.

    Also check out the parallelized version of the pipeline,
    :class:`~neuraxle.distributed.streaming.SequentialQueuedPipeline`.

    .. seealso::
        :class:`~neuraxle.pipeline.BasePipeline`,
        :class:`~neuraxle.pipeline.MiniBatchSequentialPipeline`
        :class:`~neuraxle.distributed.streaming.SequentialQueuedPipeline`
    """

    def __init__(self, steps: NamedStepsList):
        BasePipeline.__init__(self, steps=steps)

    def fit(self, data_inputs, expected_outputs=None) -> 'Pipeline':
        """
        After loading the last checkpoint, fit each pipeline steps

        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        return self.fit_data_container(DACT(di=data_inputs, eo=expected_outputs))

    def transform(self, data_inputs: Any):
        """
        Transform each pipeline steps with the pipe and filter design pattern.

        :param data_inputs: the data input to transform
        :return: transformed data inputs
        """
        data_container = self.transform_data_container(DACT(di=data_inputs))
        return data_container.data_inputs

    def fit_transform(self, data_inputs, expected_outputs=None) -> Tuple['Pipeline', Any]:
        """
        After loading the last checkpoint, fit transform each pipeline steps

        :param data_inputs: the data input to fit on
        :param expected_outputs: the expected data output to fit on
        :return: the pipeline itself
        """
        new_self, data_container = self.fit_transform_data_container(
            DACT(di=data_inputs, eo=expected_outputs))
        return new_self, data_container.data_inputs

    def inverse_transform(self, processed_outputs) -> Any:
        """
        After transforming all data inputs, and obtaining a prediction, we can inverse transform the processed outputs

        :param processed_outputs: the forward transformed data input
        :return: backward transformed processed outputs
        """
        warn_deprecated_arg(self, "handler method", "handle_inverse_transform",
                            "inverse_transform", "handle_inverse_transform", Tuple[DACT, CX])
        context = CX(execution_mode=ExecutionMode.INVERSE_TRANSFORM)
        data_container = DACT(data_inputs=processed_outputs)
        for step in list(reversed(self.values())):
            data_container = step.handle_inverse_transform(data_container, context)
        return data_container.data_inputs

    def fit_data_container(self, data_container) -> 'Pipeline':
        context = CX(execution_mode=ExecutionMode.FIT)
        return self.handle_fit(data_container, context)

    def transform_data_container(self, data_container: DACT):
        context = CX(execution_mode=ExecutionMode.TRANSFORM)
        data_container = self.handle_transform(data_container, context)
        return data_container

    def fit_transform_data_container(self, data_container) -> Tuple['Pipeline', DACT]:
        context = CX(execution_mode=ExecutionMode.FIT_TRANSFORM)
        new_self, data_container = self.handle_fit_transform(data_container, context)
        return new_self, data_container

    def _fit_data_container(self, data_container: DACT, context: CX) -> 'Pipeline':
        """
        After loading the last checkpoint, fit transform each pipeline steps,
        but only fit the last pipeline step.

        :param data_container: the data container to fit transform on
        :param context: execution context
        :return: tuple(pipeline, data_container)
        """
        index_last_step = len(self.steps) - 1
        new_steps_as_tuple: NamedStepsList = []

        for index, (step_name, step) in enumerate(self.steps_as_tuple):
            if index != index_last_step:
                step, data_container = step.handle_fit_transform(data_container, context)
            else:
                step = step.handle_fit(data_container, context)

            new_steps_as_tuple.append((step_name, step))
        self.set_steps(new_steps_as_tuple)
        return self

    def _transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        """
        After loading the last checkpoint, transform each pipeline steps

        :param data_container: the data container to transform
        :return: transformed data container
        """
        for _, step in self.steps_as_tuple:
            data_container = step.handle_transform(data_container, context)
        return data_container

    def _fit_transform_data_container(
        self, data_container: DACT, context: CX
    ) -> Tuple['Pipeline', DACT]:
        """
        After loading the last checkpoint, fit transform each pipeline steps

        :param data_container: the data container to fit transform on
        :param context: execution context
        :return: tuple(pipeline, data_container)
        """
        new_steps_as_tuple: NamedStepsList = []

        for step_name, step in self.steps_as_tuple:
            step, data_container = step.handle_fit_transform(data_container, context)
            new_steps_as_tuple.append((step_name, step))

        self.set_steps(new_steps_as_tuple)
        return self, data_container

    def _inverse_transform_data_container(
            self, data_container: DACT, context: CX) -> DACT:
        """
        After transforming all data inputs, and obtaining a prediction, we can inverse transform the processed outputs
        """
        for step_name, step in list(reversed(self.items())):
            data_container = step.handle_inverse_transform(data_container, context)
        return data_container


class MiniBatchSequentialPipeline(_TruncableServiceWithBodyMixin, _CustomHandlerMethods, ForceHandleMixin, Pipeline):
    """
    Mini Batch Sequential Pipeline class to create a pipeline processing data inputs in batch.

    Provide a default batch size :

    .. code-block:: python

        data_inputs = list(range(10))
        pipeline = MiniBatchSequentialPipeline([
            SomeStep()
        ], batch_size=2)
        pipeline.transform(data_inputs)
        # SomeStep will receive: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

        pipeline = MiniBatchSequentialPipeline([
            SomeStep()
        ], batch_size=3, keep_incomplete_batch=False)
        pipeline.transform(data_inputs)
        # SomeStep will receive: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        pipeline = MiniBatchSequentialPipeline(
            [SomeStep()],
            batch_size=3,
            keep_incomplete_batch=True,
            default_value_data_inputs=None,
            default_value_expected_outputs=None
        )
        pipeline.transform(data_inputs)
        # SomeStep will receive: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]

        pipeline = MiniBatchSequentialPipeline(
            [SomeStep()],
            batch_size=3,
            keep_incomplete_batch=True,
            default_value_data_inputs=StripAbsentValues()
        )
        pipeline.transform(data_inputs)
        # SomeStep will receive: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]


    Or manually add one or multiple :class`Barrier` steps to the mini batch sequential pipeline :

    .. code-block:: python

        data_inputs = list(range(10))
        pipeline = MiniBatchSequentialPipeline([
            SomeStep(),
            Joiner(batch_size=2)
        ])
        pipeline.transform(data_inputs)
        # SomeStep will receive: [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]

        pipeline = MiniBatchSequentialPipeline([
            SomeStep(),
            Joiner(batch_size=3, keep_incomplete_batch=False)
        ])
        pipeline.transform(data_inputs)
        # SomeStep will receive: [[0, 1, 2], [3, 4, 5], [6, 7, 8]]

        pipeline = MiniBatchSequentialPipeline([
            SomeStep(),
            Joiner(
                batch_size=3,
                keep_incomplete_batch=True,
                default_value_data_inputs=None,
                default_value_expected_outputs=None
            )
        ])
        pipeline.transform(data_inputs)
        # SomeStep will receive: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, None, None]]

        pipeline = MiniBatchSequentialPipeline([
            SomeStep(),
            Joiner(
                batch_size=3,
                keep_incomplete_batch=True,
                default_value_data_inputs=StripAbsentValues()
            )
        ])
        pipeline.transform(data_inputs)
        # SomeStep will receive: [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]

    Note that the default value for non-striped ids is None if not stripping incomplete batches of data inputs or expected outputs.

    :param steps: pipeline steps
    :param batch_size: number of elements to combine into a single batch
    :param keep_incomplete_batch: (Optional.) A bool representing
    whether the last batch should be dropped in the case it has fewer than
    `batch_size` elements; the default behavior is not to drop the smaller
    batch.
    :param default_value_data_inputs: expected_outputs default fill value
    for padding and values outside iteration range, or :class:`~neuraxle.data_container.StripAbsentValues`
    to trim absent values from the batch
    :param default_value_expected_outputs: expected_outputs default fill value
    for padding and values outside iteration range, or :class:`~neuraxle.data_container.StripAbsentValues`
    to trim absent values from the batch
    :param cache_folder: cache_folder if its at the root of the pipeline
    :param mute_joiner_batch_size_warning: If False, will log a warning when automatically setting the joiner batch_size attribute.

    .. seealso::
        :class:`~neuraxle.data_container.DataContainer`,
        :func:`~neuraxle.data_container.DataContainer.minibatches`,
        :class:`~neuraxle.data_container.StripAbsentValues`,
        :class:`Pipeline`,
        :class:`Barrier`,
        :class:`Joiner`,
        :class:`~neuraxle.base.ExecutionContext`,
        :class:`~neuraxle.distributed.streaming.SequentialQueuedPipeline`

    """

    def __init__(
            self,
            steps: NamedStepsList,
            batch_size=None,
            keep_incomplete_batch: bool = None,
            default_value_data_inputs=StripAbsentValues(),
            default_value_expected_outputs=None,
            mute_joiner_batch_size_warning: bool = True
    ):
        Pipeline.__init__(self, steps=steps)
        ForceHandleMixin.__init__(self)
        _CustomHandlerMethods.__init__(self)
        _TruncableServiceWithBodyMixin.__init__(self)

        self.default_value_data_inputs = default_value_data_inputs
        self.default_value_expected_outputs = default_value_expected_outputs
        self._validate_barriers_batch_size(batch_size=batch_size)
        self._patch_missing_barrier(
            batch_size=batch_size,
            keep_incomplete_batch=keep_incomplete_batch,
            default_value_data_inputs=default_value_data_inputs,
            default_value_expected_outputs=default_value_expected_outputs
        )
        self.mute_joiner_batch_size_warning = mute_joiner_batch_size_warning
        self._patch_barriers_batch_size(batch_size)

    def set_batch_size(self, batch_size):
        self._patch_barriers_batch_size(batch_size)

    def _validate_barriers_batch_size(self, batch_size):
        if batch_size is not None:
            return

        for _, step in self:
            if isinstance(step, Barrier):
                if step.batch_size is None:
                    raise Exception(
                        'Invalid Joiner batch size {}[{}]. Please provide a default batch size to MiniBatchSequentialPipeline, or add a batch size to {}[{}].'.format(
                            self.name, step.name, self.name, step.name))

    def _patch_barriers_batch_size(self, batch_size):
        if batch_size is None:
            return

        for _, step in self:
            if isinstance(step, Barrier):
                if step.batch_size is not None and not self.mute_joiner_batch_size_warning:
                    warnings.warn(
                        'Replacing {}[{}].batch_size by {}.batch_size.'.format(self.name, step.name, self.name))
                step.batch_size = batch_size

    def _patch_missing_barrier(
            self,
            batch_size: int,
            keep_incomplete_batch: bool,
            default_value_data_inputs: Union[Any, StripAbsentValues] = None,
            default_value_expected_outputs: Union[Any, StripAbsentValues] = None
    ):
        has_barrier: bool = False

        for _, step in self:
            if isinstance(step, Barrier):
                has_barrier = True

        if not has_barrier:
            self.steps_as_tuple.append((
                'Joiner',
                Joiner(
                    batch_size=batch_size,
                    keep_incomplete_batch=keep_incomplete_batch,
                    default_value_data_inputs=default_value_data_inputs,
                    default_value_expected_outputs=default_value_expected_outputs
                )
            ))

        self._refresh_steps()

    def transform_data_container(self, data_container: DACT, context: CX) -> DACT:
        """
        Transform all sub pipelines splitted by the Barrier steps.
        :param data_container: data container to transform.
        :param context: execution context
        :return: data container
        """
        sub_pipelines: List['MiniBatchSequentialPipeline'] = self._split_on_barriers()

        for sub_pipeline in sub_pipelines:
            barrier: Barrier = sub_pipeline.joiner
            data_container = barrier.join_transform(
                step=sub_pipeline,
                data_container=data_container,
                context=context
            )

        return data_container

    def fit_data_container(self, data_container: DACT, context: CX) -> BaseStep:
        """
        Fit all sub pipelines splitted by the Barrier steps.
        :param data_container: data container to transform.
        :param context: execution context
        :return: data container
        """
        self, data_container = self.fit_transform_data_container(data_container, context)
        return self

    def fit_transform_data_container(self, data_container: DACT, context: CX) -> Tuple[
            BaseStep, DACT]:
        """
        Transform all sub pipelines splitted by the Barrier steps.
        :param data_container: data container to transform.
        :param context: execution context
        :return: data container
        """
        sub_pipelines: List['MiniBatchSequentialPipeline'] = self._split_on_barriers()
        index_start = 0

        for sub_pipeline in sub_pipelines:
            sub_pipeline._setup(context=context)

            barrier: Barrier = sub_pipeline.joiner
            sub_pipeline, data_container = barrier.join_fit_transform(
                step=sub_pipeline,
                data_container=data_container,
                context=context
            )

            new_self = self[:index_start] + sub_pipeline
            if index_start + len(sub_pipeline) < len(self):
                new_self += self[index_start + len(sub_pipeline):]

            self.steps_as_tuple = new_self.steps_as_tuple
            index_start += len(sub_pipeline)

        return self, data_container

    def _split_on_barriers(self) -> List['MiniBatchSequentialPipeline']:
        """
        Create sub pipelines by splitting the steps by the join type name.
        :return: list of sub pipelines
        """
        sub_pipelines: List[MiniBatchSequentialPipeline] = self.split(Barrier)
        for sub_pipeline in sub_pipelines:
            if not sub_pipeline.ends_with(Barrier):
                raise Exception('At least one Barrier/Joiner step needs to be at the end of a streaming pipeline.')

        return sub_pipelines


class Barrier(Identity, ABC):
    """
    A Barrier step to be used in a minibatch sequential pipeline. It forces all the
    data inputs to get to the barrier in a sub pipeline before going through to the next sub-pipeline.

    .. code-block:: python

        p = MiniBatchSequentialPipeline([
            SomeStep(),
            SomeStep(),
            Barrier(), # must be a concrete Barrier ex: Joiner()
            SomeStep(),
            SomeStep(),
            Barrier(), # must be a concrete Barrier ex: Joiner()
        ], batch_size=10)


    .. seealso::
        :class:`~neuraxle.base.NonTransformableMixin`,
        :class:`~neuraxle.base.BaseStep`
    """

    @abstractmethod
    def join_transform(self, step: TruncableSteps, data_container: DACT,
                       context: CX) -> DACT:
        """
        Execute the given pipeline :func:`~neuraxle.pipeline.Pipeline.transform` with the given data container, and execution context.
        :param step: truncable steps to execute
        :type step: TruncableSteps
        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: transformed data container
        :rtype: DataContainer
        """
        raise NotImplementedError()

    @abstractmethod
    def join_fit_transform(self, step: Pipeline, data_container: DACT, context: CX) -> Tuple[
            'Any', DACT]:
        """
        Execute the given pipeline :func:`~neuraxle.pipeline.Pipeline.fit_transform` with the given data container, and execution context.
        :param step: truncable steps to execute
        :param data_container: data container
        :param context: execution context
        :return: (fitted step, transformed data container)
        """
        raise NotImplementedError()


class Joiner(Barrier):
    """
    The Joiner step joins the transformed mini batches together with
    mostly the DACT.minibatches and then DACT.extend method.
    It is used in a minibatch sequential pipeline and streaming / queued pipeline
    as a way to handle batches of previous steps in the pipeline.

    .. seealso::
        :class:`~neuraxle.data_container.DataContainer`,
        :func:`~neuraxle.data_container.DataContainer.batch`
    """

    def __init__(
            self,
            batch_size: int,
            keep_incomplete_batch: bool = True,
            default_value_data_inputs=StripAbsentValues(),
            default_value_expected_outputs=None
    ):
        """
        The Joiner step joins the transformed mini batches together with DACT.minibatches and then DACT.extend method.
        Note that the default value for IDs is None.

        .. seealso::
            :class:`~neuraxle.data_container.DataContainer`,
            :func:`~neuraxle.data_container.DataContainer.minibatches`
        """
        Barrier.__init__(self)
        self.batch_size: int = batch_size
        self.keep_incomplete_batch: bool = keep_incomplete_batch
        self.default_value_data_inputs: Union[Any, StripAbsentValues] = default_value_data_inputs
        self.default_value_expected_outputs: Union[Any, StripAbsentValues] = default_value_expected_outputs

    def join_transform(self, step: Pipeline, data_container: DACT, context: CX) -> DACT:
        """
        Concatenate the pipeline transform output of each batch of self.batch_size together.
        :param step: pipeline to transform on
        :type step: Pipeline
        :param data_container: data container to transform
        :type data_container: DataContainer
        :param context: execution context
        :return: transformed data container
        :rtype: DataContainer
        """
        context = context.push(step)
        dact_batches = data_container.minibatches(
            batch_size=self.batch_size,
            keep_incomplete_batch=self.keep_incomplete_batch,
            default_value_data_inputs=self.default_value_data_inputs,
            default_value_expected_outputs=self.default_value_expected_outputs
        )

        out_dact = ListDataContainer.empty()
        for dact_batch in dact_batches:
            processed_dact_batch = step._transform_data_container(dact_batch, context)
            out_dact.extend(processed_dact_batch)

        return out_dact

    def join_fit_transform(self, step: Pipeline, data_container: DACT, context: CX) -> \
            Tuple['Any', DACT]:
        """
        Concatenate the pipeline fit transform output of each batch of self.batch_size together.
        :param step: pipeline to fit transform on
        :type step: Pipeline
        :param data_container: data container to fit transform on
        :type data_container: DataContainer
        :param context: execution context
        :return: fitted self, transformed data inputs
        :rtype: Tuple[Any, DataContainer]
        """
        context = context.push(step)
        dact_batches = data_container.minibatches(
            batch_size=self.batch_size,
            keep_incomplete_batch=self.keep_incomplete_batch,
            default_value_data_inputs=self.default_value_data_inputs,
            default_value_expected_outputs=self.default_value_expected_outputs
        )

        out_dact = ListDataContainer.empty()
        for dact_batch in dact_batches:
            step, dact_batch = step._fit_transform_data_container(dact_batch, context)
            out_dact.extend(dact_batch)

        return step, out_dact


class ZipMinibatchJoiner(Joiner):
    """
    Zips together minibatch outputs, i.e. returns a DataContainer where the first
    element is a tuple of every minibatches first element and so on.
    """

    def join_transform(self, step: TruncableSteps, data_container: DACT,
                       context: CX) -> ZipDataContainer:
        context = context.push(step)
        data_container_batches = data_container.minibatches(
            batch_size=self.batch_size,
            keep_incomplete_batch=self.keep_incomplete_batch,
            default_value_data_inputs=self.default_value_data_inputs,
            default_value_expected_outputs=self.default_value_expected_outputs
        )

        output_data_container = []
        for data_container_batch in data_container_batches:
            output_data_container.append(step._transform_data_container(data_container_batch, context))

        return ZipDataContainer.create_from(*output_data_container)

    def join_fit_transform(self, step: Pipeline, data_container: DACT, context: CX) -> \
            Tuple['Any', DACT]:
        context = context.push(step)
        data_container_batches = data_container.minibatches(
            batch_size=self.batch_size,
            keep_incomplete_batch=self.keep_incomplete_batch,
            default_value_data_inputs=self.default_value_data_inputs,
            default_value_expected_outputs=self.default_value_expected_outputs
        )

        output_data_container = []
        for data_container_batch in data_container_batches:
            step, data_container_batch = step._fit_transform_data_container(data_container_batch, context)
            output_data_container.append(data_container_batch)

        return step, ZipDataContainer.create_from(*output_data_container)
