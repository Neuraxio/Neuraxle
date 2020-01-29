"""
Pipeline Steps For Looping
=====================================

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
import copy
from typing import List, Any

import numpy as np

from neuraxle.base import MetaStepMixin, BaseStep, DataContainer, ExecutionContext, ResumableStepMixin
from neuraxle.data_container import ListDataContainer
from neuraxle.hyperparams.space import HyperparameterSamples, HyperparameterSpace


class ForEachDataInput(ResumableStepMixin, MetaStepMixin, BaseStep):
    """
    Truncable step that fits/transforms each step for each of the data inputs, and expected outputs.

    .. seealso::
        :class:`neuraxle.base.BaseStep`,
        :class:`neuraxle.base.BaseSaver`,
        :class:`neuraxle.base.BaseHasher`,
        :class:`neuraxle.base.ResumableStepMixin`,
        :class:`neuraxle.base.NonFittableMixin`,
        :class:`neuraxle.base.NonTransformableMixin`,
        :class:`neuraxle.pipeline.Pipeline`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`neuraxle.data_container.DataContainer`
    """

    def __init__(
            self,
            wrapped: BaseStep
    ):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)

    def fit(self, data_inputs, expected_outputs=None):
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)

        for di, eo in zip(data_inputs, expected_outputs):
            self.wrapped = self.wrapped.fit(di, eo)

        return self

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> BaseStep:
        """
        Fit each step for each data inputs, and expected outputs

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: self
        """
        for current_id, di, eo in data_container:
            self.wrapped = self.wrapped.handle_fit(
                DataContainer(data_inputs=di, current_ids=None, expected_outputs=eo),
                context
            )

        return self

    def transform(self, data_inputs):
        """
        Transform each step for each data inputs.

        :param data_inputs: data inputs to transform
        :type data_inputs: Iterable
        :return: outputs
        """
        outputs = []
        for di in data_inputs:
            outputs.append(self.wrapped.transform(di))

        return outputs

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        """
        Transform each step for each data inputs.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: self
        """
        output_data_container = ListDataContainer.empty()

        for current_id, di, eo in data_container:
            output = self.wrapped.handle_transform(
                DataContainer(data_inputs=di, current_ids=None, expected_outputs=eo),
                context
            )

            output_data_container.append(
                current_id,
                output.data_inputs,
                output.expected_outputs
            )
        output_data_container.summary_id = data_container.summary_id

        return output_data_container

    def fit_transform(self, data_inputs, expected_outputs=None):
        """
        Fit transform each step for each data inputs, and expected outputs

        :param data_inputs: data inputs to fit transform
        :type data_inputs: Iterable
        :param expected_outputs: expected outputs to fit transform on
        :type expected_outputs: Iterable

        :return: self, transformed_data_container
        """
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)

        outputs = []
        for di, eo in zip(data_inputs, expected_outputs):
            self.wrapped, output = self.wrapped.fit_transform(di, eo)
            outputs.append(output)

        return self

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext):
        """
        Fit transform each step for each data inputs, and expected outputs

        :param data_container: data container to fit transform
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext

        :return: self, transformed_data_container
        """
        output_data_container = ListDataContainer.empty()

        for current_id, di, eo in data_container:
            self.wrapped, output = self.wrapped.handle_fit_transform(
                DataContainer(data_inputs=di, current_ids=None, expected_outputs=eo),
                context
            )

            output_data_container.append(
                current_id,
                output.data_inputs,
                output.expected_outputs
            )

        output_data_container.summary_id = data_container.summary_id

        return self, output_data_container

    def hash_data_container(self, data_container):
        output_data_container = self.wrapped.hash_data_container(data_container)
        output_data_container.summary_id = data_container.summary_id

        return output_data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext):
        context = context.push(self)

        if isinstance(self.wrapped, ResumableStepMixin) and self.wrapped.should_resume(data_container, context):
            return True
        return False


class StepClonerForEachDataInput(MetaStepMixin, BaseStep):
    def __init__(self, wrapped: BaseStep, copy_op=copy.deepcopy):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)

        self.set_step(wrapped)
        self.steps: List[BaseStep] = []
        self.copy_op = copy_op

    def set_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        MetaStepMixin.set_hyperparams(self, hyperparams)
        self.steps = [s.set_hyperparams(self.wrapped.get_hyperparams()) for s in self.steps]
        return self

    def update_hyperparams(self, hyperparams: HyperparameterSamples) -> BaseStep:
        """
        Update the step hyperparameters without removing the already-set hyperparameters.
        Please refer to :func:`~BaseStep.update_hyperparams`.

        :param hyperparams: hyperparams to update
        :type hyperparams: HyperparameterSamples
        :return: self
        :rtype: BaseStep

        .. seealso::
            :func:`~BaseStep.update_hyperparams`,
            :class:`HyperparameterSamples`
        """
        MetaStepMixin.update_hyperparams(self, hyperparams)
        self.steps = [s.set_hyperparams(self.wrapped.get_hyperparams()) for s in self.steps]
        return self

    def set_hyperparams_space(self, hyperparams_space: HyperparameterSpace) -> 'BaseStep':
        MetaStepMixin.set_hyperparams_space(self, hyperparams_space)
        self.steps = [s.set_hyperparams_space(self.wrapped.get_hyperparams_space()) for s in self.steps]
        return self

    def fit_transform(self, data_inputs, expected_outputs=None) -> ('BaseStep', Any):
        # One copy of step per data input:
        self.steps = [self.copy_op(self.wrapped) for _ in range(len(data_inputs))]

        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)

        fit_transform_result = [self.steps[i].fit_transform(di, eo) for i, (di, eo) in
                                enumerate(zip(data_inputs, expected_outputs))]
        self.steps = [step for step, di in fit_transform_result]
        data_inputs = [di for step, di in fit_transform_result]

        return self, data_inputs

    def fit(self, data_inputs: List, expected_outputs: List = None) -> 'StepClonerForEachDataInput':
        # One copy of step per data input:
        self.steps = [self.copy_op(self.wrapped) for _ in range(len(data_inputs))]

        # Fit them all.
        if expected_outputs is None:
            expected_outputs = [None] * len(data_inputs)
        self.steps = [self.steps[i].fit(di, eo) for i, (di, eo) in enumerate(zip(data_inputs, expected_outputs))]

        return self

    def transform(self, data_inputs: List) -> List:
        return [self.steps[i].transform(di) for i, di in enumerate(data_inputs)]

    def inverse_transform(self, data_output):
        return [self.steps[i].inverse_transform(di) for i, di in enumerate(data_output)]


class FlattenForEach(ResumableStepMixin, MetaStepMixin, BaseStep):
    """
    Step that reduces a dimension instead of manually looping on it.

    .. seealso::
        :class:`neuraxle.base.BaseStep`,
        :class:`neuraxle.base.BaseSaver`,
        :class:`neuraxle.base.BaseHasher`,
        :class:`neuraxle.base.ResumableStepMixin`,
        :class:`neuraxle.base.MetaStepMixin`,
        :class:`neuraxle.base.NonFittableMixin`,
        :class:`neuraxle.base.NonTransformableMixin`,
        :class:`neuraxle.pipeline.Pipeline`,
        :class:`neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`neuraxle.data_container.DataContainer`
    """

    def __init__(
            self,
            wrapped: BaseStep,
            reaugment: bool = True
    ):
        BaseStep.__init__(self)
        MetaStepMixin.__init__(self, wrapped)
        ResumableStepMixin.__init__(self)

        self.reaugment = reaugment

    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        """
        Flatten data container before any processing is done on the wrapped step.

        :param data_container: data container to flatten
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: (data container, execution context)
        :rtype: ('BaseStep', DataContainer)
        """
        data_container, context = BaseStep._will_process(self, data_container, context)

        if data_container.expected_outputs is None or not np.all(np.array(data_container.expected_outputs)):
            expected_outputs = np.empty_like(np.array(data_container.data_inputs))
            expected_outputs.fill(np.nan)
            data_container.set_expected_outputs(expected_outputs)

        self.flattened_dimension_lengths = [len(di) for di in data_container.data_inputs]

        data_container.set_data_inputs(self._flatten_list(data_container.data_inputs))
        data_container.set_expected_outputs(self._flatten_list(data_container.expected_outputs))

        return data_container, context

    def _flatten_list(self, list_to_flatten):
        """
        Flatten the first dimension of a list.

        :param list_to_flatten: list to flatten
        :type list_to_flatten: Iterable
        :return: flattened list
        :rtype: np.ndarray
        """
        if not isinstance(list_to_flatten, np.ndarray):
            list_to_flatten = np.array(list_to_flatten)

        return np.array(sum(list_to_flatten.tolist(), []))

    def _did_process(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Reaugment the flattened data container.

        :param data_container: data container to reaugment
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: data container
        :rtype: DataContainer
        """
        data_container = BaseStep._did_process(self, data_container, context)

        if self.reaugment:
            data_container.set_data_inputs(self._reaugment_list(data_container.data_inputs))
            data_container.set_expected_outputs(self._reaugment_list(data_container.expected_outputs))

        return data_container

    def _reaugment_list(self, list_to_reaugment):
        """
        Reaugment list with the flattened dimension lengths.

        :param list_to_reaugment: list to reaugment
        :type list_to_reaugment: Iterable
        :return: reaugmented numpy array
        :rtype: np.ndarray
        """
        if not self.reaugment:
            return list_to_reaugment

        reaugmented_list = []
        i = 0
        for list_length in self.flattened_dimension_lengths:
            sub_list = np.array(list_to_reaugment[i:i + list_length]).tolist()
            reaugmented_list.append(sub_list)
            i += list_length

        reaugmented_list = np.array(reaugmented_list)
        return reaugmented_list

    def should_resume(self, data_container: DataContainer, context: ExecutionContext):
        context = context.push(self)

        if isinstance(self.wrapped, ResumableStepMixin) and self.wrapped.should_resume(data_container, context):
            return True
        return False
