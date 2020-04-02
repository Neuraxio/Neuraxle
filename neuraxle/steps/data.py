"""
Data Steps
====================================
You can find here steps that take action on data.

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
import random
from typing import Iterable

from neuraxle.base import BaseStep, MetaStepMixin, NonFittableMixin, ExecutionContext, HandleOnlyMixin, \
    ForceHandleOnlyMixin
from neuraxle.base import NonTransformableMixin
from neuraxle.data_container import DataContainer, _inner_concatenate_np_array
from neuraxle.pipeline import Pipeline
from neuraxle.steps.flow import TrainOnlyWrapper
from neuraxle.steps.output_handlers import InputAndOutputTransformerMixin


class DataShuffler(NonFittableMixin, InputAndOutputTransformerMixin, BaseStep):
    """
    Data Shuffling step that shuffles data inputs, and expected_outputs at the same time.

    .. code-block:: python

        p = Pipeline([
            TrainOnlyWrapper(DataShuffler(seed=42, increment_seed_after_each_fit=True, increment_seed_after_each_fit=False)),
            EpochRepeater(ForecastingPipeline(), epochs=EPOCHS, repeat_in_test_mode=False)
        ])

    .. warning::
        You probably always want to wrap this step by a :class:`TrainOnlyWrapper`

    .. seealso::
        :class:`EpochRepeater`,
        :class:`~neuraxle.steps.flow.TrainOnlyWrapper`,
        :class:`~neuraxle.steps.output_handlers.InputAndOutputTransformerMixin`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self, seed=None, increment_seed_after_each_fit=True):
        InputAndOutputTransformerMixin.__init__(self)
        BaseStep.__init__(self)
        if seed is None:
            seed = 42
        self.seed = seed
        self.increment_seed_after_each_fit = increment_seed_after_each_fit

    def transform(self, data_inputs):
        """
        Shuffle data inputs, and expected outputs.

        :param data_inputs: (data inputs, expected outputs) tuple to shuffle
        :return:
        """
        if self.increment_seed_after_each_fit:
            self.seed += 1

        di, eo = data_inputs
        data = list(zip(di, eo))
        random.Random(self.seed).shuffle(data)

        data_inputs_shuffled, expected_outputs_shuffled = list(zip(*data))

        return list(data_inputs_shuffled), list(expected_outputs_shuffled)


class EpochRepeater(ForceHandleOnlyMixin, MetaStepMixin, BaseStep):
    """
    Repeat wrapped step fit, or transform for the number of epochs passed in the constructor.

    .. code-block:: python

        p = Pipeline([
            TrainOnlyWrapper(DataShuffler(seed=42, increment_seed_after_each_fit=True, increment_seed_after_each_fit=False)),
            EpochRepeater(ForecastingPipeline(), epochs=EPOCHS, repeat_in_test_mode=False)
        ])

    .. seealso::
        :class:`DataShuffler`,
        :class:`~neuraxle.base.MetaStepMixin`,
        :class:`~neuraxle.steps.flow.TrainOnlyWrapper`,
        :class:`~neuraxle.steps.flow.TestOnlyWrapper`,
        :class:`~neuraxle.base.BaseStep`
    """

    def __init__(self, wrapped, epochs, repeat_in_test_mode=False, cache_folder_when_no_handle=None):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        ForceHandleOnlyMixin.__init__(self, cache_folder=cache_folder_when_no_handle)

        self.repeat_in_test_mode = repeat_in_test_mode
        self.epochs = epochs

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Fit transform wrapped step self.epochs times using wrapped step handle fit transform method.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: (fitted self, data container)
        :rtype: (BaseStep, DataContainer)
        """
        if self._should_repeat():
            for _ in range(self.epochs - 1):
                self.wrapped = self.wrapped.handle_fit(data_container.copy(), context)

        self.wrapped, data_container = self.wrapped.handle_fit_transform(data_container, context)
        return self, data_container

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Iterable):
        """
        Fit transform wrapped step self.epochs times.

        :param data_inputs: data inputs to fit on
        :param expected_outputs: expected_outputs to fit on
        :return: fitted self
        """
        if self._should_repeat():
            for _ in range(self.epochs - 1):
                self.wrapped = self.wrapped.fit(data_inputs, expected_outputs)

        self.wrapped, outputs = self.wrapped.fit_transform(data_inputs, expected_outputs)

        return self, outputs

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> 'BaseStep':
        """
        Fit wrapped step self.epochs times using wrapped step handle fit method.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: (fitted self, data container)
        :rtype: (BaseStep, DataContainer)
        """

        if self._should_repeat():
            for _ in range(self.epochs):
                self.wrapped = self.wrapped.handle_fit(data_container.copy(), context)
        return self

    def fit(self, data_inputs, expected_outputs=None) -> 'BaseStep':
        """
        Fit wrapped step self.epochs times.

        :param data_inputs: data inputs to fit on
        :param expected_outputs: expected_outputs to fit on
        :return: fitted self
        """
        if self._should_repeat():
            for _ in range(self.epochs):
                self.wrapped = self.wrapped.fit(data_inputs, expected_outputs)
        else:
            self.wrapped = self.wrapped.fit(data_inputs, expected_outputs)

        return self

    def _should_repeat(self):
        return self.is_train or (not self.is_train and self.repeat_in_test_mode)

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        return self.wrapped.handle_transform(data_container, context)

    def _get_epochs(self):
        epochs = self.epochs
        if not self._should_repeat_fit():
            epochs = 1
        return epochs

    def _should_repeat_fit(self):
        return self.is_train or (not self.is_train and self.repeat_in_test_mode)


class TrainShuffled(Pipeline):
    def __init__(self, wrapped, seed=None):
        Pipeline.__init__(self, [
            TrainOnlyWrapper(DataShuffler(seed=seed)),
            wrapped
        ])


class InnerConcatenateDataContainer(NonFittableMixin, NonTransformableMixin, BaseStep):
    """
    Concatenate inner features of sub data containers along `axis=-1`..

    Code example:

    .. code-block:: python

        data_container = DataContainer(data_inputs=data_inputs_3d, expected_outputs=expected_outputs_3d)
        data_container.add_sub_data_container(name='1d_data_source', data_container=data_container_1d)
        data_container.add_sub_data_container(name='2d_data_source', data_container=data_container_2d)

        # data container with sub data containers :
        # DataContainer(data_inputs=data_inputs_3d, expected_outputs=expected_outputs, sub_data_containers=[('1d_data_source', data_container_1d), ('2d_data_source', data_container_2d)])

        p = Pipeline([
            InnerConcatenateDataContainer()
            # is equivalent to ZipData(sub_data_container_names=['1d_data_source', '2d_data_source'])
        ])

        data_container = p.handle_transform(data_container, ExecutionContext())

        # new_shape: (batch_size, time_steps, n_features + batch_features + 1)


    .. seealso::
        :class:`~neuraxle.base.NonFittableMixin`,
        :class:`~neuraxle.base.NonTransformableMixin`,
        :class:`~neuraxle.base.BaseStep`
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, sub_data_container_names=None):
        BaseStep.__init__(self)
        NonTransformableMixin.__init__(self)
        NonFittableMixin.__init__(self)

        self.data_sources = sub_data_container_names

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Merge sub data containers into the current data container.

        :param data_container: data container to zip
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: base step, data container
        :rtype: Tuple[BaseStep, DataContainer]
        """
        return self, self._concatenate_sub_data_containers(data_container)

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Merge sub data containers into the current data container.

        :param data_container: data container to zip
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: base step, data container
        :rtype: DataContainer
        """
        return self._concatenate_sub_data_containers(data_container)

    def _concatenate_sub_data_containers(self, data_container: DataContainer):
        """
        Merge sub data containers into the current data container.

        :param data_container: data container to zip
        :type data_container: DataContainer
        :return: base step, data container
        :rtype: DataContainer
        """
        sub_data_containers_to_zip = []
        if self.data_sources is None:
            self.data_sources = data_container.get_sub_data_container_names()

        for name, sub_data_container in data_container.sub_data_containers:
            if name in self.data_sources:
                sub_data_containers_to_zip.append(sub_data_container)

        for data_container_to_zip in sub_data_containers_to_zip:
            data_container = self._concatenate_sub_data_container(data_container, data_container_to_zip)

        return data_container

    def _concatenate_sub_data_container(self, data_container, data_container_to_zip) -> DataContainer:
        """
        Zip a data container into another data container with a higher dimension.

        :param data_container: data container
        :type data_container: DataContainer
        :param data_container_to_zip: data container to concatenate
        :type data_container_to_zip: DataContainer
        :return: concatenated data containers
        """
        data_inputs = _inner_concatenate_np_array(data_container.data_inputs,
                                                  data_container_to_zip.data_inputs)
        data_container.set_data_inputs(data_inputs)

        expected_outputs = _inner_concatenate_np_array(data_container.expected_outputs,
                                                       data_container_to_zip.expected_outputs)
        data_container.set_expected_outputs(expected_outputs)

        return data_container


class ZipBatchDataContainer(NonFittableMixin, NonTransformableMixin, BaseStep):
    """
    Concatenate outer batch of sub data containers along `axis=0`..

    Code example:

    .. code-block:: python

        data_container = DataContainer(data_inputs=data_inputs_3d, expected_outputs=expected_outputs_3d)
        data_container.add_sub_data_container(name='1d_data_source', data_container=data_container_1d)
        data_container.add_sub_data_container(name='2d_data_source', data_container=data_container_2d)

        # data container with sub data containers :
        # DataContainer(data_inputs=data_inputs_3d, expected_outputs=expected_outputs, sub_data_containers=[('1d_data_source', data_container_1d), ('2d_data_source', data_container_2d)])

        p = Pipeline([
            ZipBatchDataContainer()
            # is equivalent to ZipBatchDataContainer(sub_data_container_names=['2d_data_source'])
        ])

        data_container = p.handle_transform(data_container, ExecutionContext())

        # new_shape: (batch_size, ((time_steps, n_features_3d), n_features_2d))


    .. seealso::
        :class:`~neuraxle.base.NonFittableMixin`,
        :class:`~neuraxle.base.NonTransformableMixin`,
        :class:`~neuraxle.base.BaseStep`
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, sub_data_container_names=None):
        BaseStep.__init__(self)
        NonTransformableMixin.__init__(self)
        NonFittableMixin.__init__(self)

        self.data_sources = sub_data_container_names

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Merge sub data containers into the current data container.

        :param data_container: data container to zip
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: base step, data container
        :rtype: Tuple[BaseStep, DataContainer]
        """
        return self, self._batch_zip_sub_data_containers(data_container)

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Merge sub data containers into the current data container.

        :param data_container: data container to zip
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: base step, data container
        :rtype: DataContainer
        """
        return self._batch_zip_sub_data_containers(data_container)

    def _batch_zip_sub_data_containers(self, data_container: DataContainer):
        """
        Zip sub data containers on the batch dimension.

        :param data_container: data container to zip
        :type data_container: DataContainer
        :return: base step, data container
        :rtype: DataContainer
        """
        sub_data_containers_to_zip = []
        if self.data_sources is None:
            self.data_sources = data_container.get_sub_data_container_names()

        for name, sub_data_container in data_container.sub_data_containers:
            if name in self.data_sources:
                sub_data_containers_to_zip.append(sub_data_container)

        for data_container_to_zip in sub_data_containers_to_zip:
            data_container = self._batch_zip_sub_data_container(data_container, data_container_to_zip)

        return data_container

    def _batch_zip_sub_data_container(self, data_container, data_container_to_zip) -> DataContainer:
        """
        Zip sub data container on the batch dimension.

        :param data_container: data container
        :type data_container: DataContainer
        :param data_container_to_zip: data container to concatenate
        :type data_container_to_zip: DataContainer
        :return: concatenated data containers
        """
        new_data_inputs = []
        for di, other_di in zip(data_container.data_inputs, data_container_to_zip.data_inputs):
            new_data_inputs.append((di, other_di))

        new_expected_outputs = []
        for eo, other_eo in zip(data_container.expected_outputs, data_container_to_zip.expected_outputs):
            new_expected_outputs.append((eo, other_eo))

        data_container.set_data_inputs(new_data_inputs)
        data_container.set_expected_outputs(new_expected_outputs)

        return data_container

