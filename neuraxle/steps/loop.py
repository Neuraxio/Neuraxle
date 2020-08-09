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
from typing import List
from typing import Tuple

import numpy as np

from neuraxle.base import MetaStep, BaseStep, DataContainer, ExecutionContext, ResumableStepMixin, \
    ForceHandleOnlyMixin, ForceHandleMixin, TruncableJoblibStepSaver, NamedTupleList, BaseTransformer, MetaStepMixin
from neuraxle.data_container import ListDataContainer


class ForEachDataInput(ForceHandleOnlyMixin, ResumableStepMixin, MetaStep):
    """
    Truncable step that fits/transforms each step for each of the data inputs, and expected outputs.

    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.base.BaseSaver`,
        :class:`~neuraxle.base.BaseHasher`,
        :class:`~neuraxle.base.ResumableStepMixin`,
        :class:`~neuraxle.base.NonTransformableMixin`,
        :class:`~neuraxle.pipeline.Pipeline`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(self, wrapped: BaseTransformer, cache_folder_when_no_handle=None):
        MetaStep.__init__(self, wrapped)
        ForceHandleOnlyMixin.__init__(self, cache_folder_when_no_handle)

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

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Transform each step for each data inputs.

        :param data_container: data container
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext
        :return: self
        """
        output_data_container: ListDataContainer = ListDataContainer.empty(original_data_container=data_container)

        for current_id, di, eo in data_container:
            output: DataContainer = self.wrapped.handle_transform(
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

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> Tuple[BaseStep, DataContainer]:
        """
        Fit transform each step for each data inputs, and expected outputs

        :param data_container: data container to fit transform
        :type data_container: DataContainer
        :param context: execution context
        :type context: ExecutionContext

        :return: self, transformed_data_container
        """
        output_data_container: DataContainer = ListDataContainer.empty(original_data_container=data_container)

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

    def hash_data_container(self, data_container: DataContainer) -> DataContainer:
        output_data_container = self.wrapped.hash_data_container(data_container)
        output_data_container.summary_id = data_container.summary_id

        return output_data_container

    def should_resume(self, data_container: DataContainer, context: ExecutionContext) -> bool:
        context: ExecutionContext = context.push(self)

        if isinstance(self.wrapped, ResumableStepMixin) and self.wrapped.should_resume(data_container, context):
            return True
        return False


class StepClonerForEachDataInput(ForceHandleOnlyMixin, MetaStep):
    def __init__(self, wrapped: BaseTransformer, copy_op=copy.deepcopy, cache_folder_when_no_handle=None):
        MetaStep.__init__(self, wrapped=wrapped)
        ForceHandleOnlyMixin.__init__(self, cache_folder_when_no_handle)
        self.savers.append(TruncableJoblibStepSaver())

        self.set_step(wrapped)
        self.steps_as_tuple: List[NamedTupleList] = []
        self.copy_op = copy_op

    def get_children(self) -> List[BaseStep]:
        """
        Get the list of all the children for that step.

        :return: list of children
        """
        children: List[BaseStep] = MetaStep.get_children(self)
        cloned_children = [step for _, step in self.steps_as_tuple]
        children.extend(cloned_children)
        return children

    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        data_container, context = super()._will_process(data_container, context)

        if len(self.steps_as_tuple) != len(data_container.data_inputs):
            self._copy_one_step_per_data_input(data_container)

        return data_container, context

    def _copy_one_step_per_data_input(self, data_container):
        # One copy of step per data input:
        steps = [self.copy_op(self.wrapped).set_name('{}[{}]'.format(self.wrapped.name, i)) for i in range(len(data_container))]
        self.steps_as_tuple = [(step.name, step) for step in steps]
        self._invalidate()

    def _fit_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        fitted_steps_data_containers = []
        for i, (current_ids, data_inputs, expected_outputs) in enumerate(data_container):
            fitted_step_data_container = self[i].handle_fit_transform(
                DataContainer(current_ids=current_ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )
            fitted_steps_data_containers.append(fitted_step_data_container)

        self.steps_as_tuple = [(step.name, step) for step, _ in fitted_steps_data_containers]

        output_data_container = ListDataContainer.empty()
        for _, data_container_batch in fitted_steps_data_containers:
            output_data_container.append_data_container(data_container_batch)

        return self, output_data_container

    def _fit_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        fitted_steps = []
        for i, (current_ids, data_inputs, expected_outputs) in enumerate(data_container):
            fitted_step = self[i].handle_fit(
                DataContainer(current_ids=current_ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )
            fitted_steps.append(fitted_step)

        self.steps_as_tuple = [(step.name, step) for step in fitted_steps]

        return self

    def _transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> ('BaseStep', DataContainer):
        transform_results = []
        for i, (current_ids, data_inputs, expected_outputs) in enumerate(data_container):
            transform_result = self[i].handle_transform(
                DataContainer(current_ids=current_ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )
            transform_results.append(transform_result)

        output_data_container = ListDataContainer.empty()
        for data_container_batch in transform_results:
            output_data_container.append_data_container(data_container_batch)
        return output_data_container

    def _inverse_transform_data_container(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        inverse_transform_results = []
        for i, (current_ids, data_inputs, expected_outputs) in enumerate(data_container):
            inverse_transform_result = self[i].handle_inverse_transform(
                DataContainer(current_ids=current_ids, data_inputs=data_inputs, expected_outputs=expected_outputs),
                context
            )
            inverse_transform_results.append(inverse_transform_result)

        output_data_container = ListDataContainer.empty()
        for data_container_batch in inverse_transform_results:
            output_data_container.append_data_container(data_container_batch)
        return output_data_container

    def inverse_transform(self, data_output):
        return [self[i].inverse_transform(di) for i, di in enumerate(data_output)]

    def __getitem__(self, item):
        """
        Get cloned step at the given index.

        :return: iter(self.steps_as_tuple)
        """
        return self.steps_as_tuple[item][1]

    def __iter__(self):
        """
        Iterate through the steps.

        :return: iter(self.steps_as_tuple)
        """
        return iter(self.steps_as_tuple)

    def __len__(self):
        """
        Get number of steps cloned for each data input.

        :return: len(self.steps_as_tuple)
        """
        return len(self.steps_as_tuple)


class FlattenForEach(ForceHandleMixin, ResumableStepMixin, MetaStep):
    """
    Step that reduces a dimension instead of manually looping on it.

    .. seealso::
        :class:`~neuraxle.base.BaseStep`,
        :class:`~neuraxle.base.BaseSaver`,
        :class:`~neuraxle.base.BaseHasher`,
        :class:`~neuraxle.base.ResumableStepMixin`,
        :class:`~neuraxle.base.MetaStepMixin`,
        :class:`~neuraxle.base.NonTransformableMixin`,
        :class:`~neuraxle.pipeline.Pipeline`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSamples`,
        :class:`~neuraxle.hyperparams.space.HyperparameterSpace`,
        :class:`~neuraxle.data_container.DataContainer`
    """

    def __init__(
            self,
            wrapped: BaseTransformer,
            then_unflatten: bool = True
    ):
        MetaStep.__init__(self, wrapped)
        ResumableStepMixin.__init__(self)
        ForceHandleMixin.__init__(self)

        self.then_unflatten = then_unflatten

        self.len_di = []
        self.len_eo = []

    def _will_process(self, data_container: DataContainer, context: ExecutionContext) -> (
    'BaseTransformer', DataContainer):
        """
        Flatten data container before any processing is done on the wrapped step.

        :param data_container: data container to flatten
        :param context: execution context
        :return: (data container, execution context)
        :rtype: ('BaseTransformer', DataContainer)
        """
        data_container, context = super()._will_process(data_container, context)

        if data_container.expected_outputs is None:
            expected_outputs = np.empty_like(np.array(data_container.data_inputs))
            expected_outputs.fill(np.nan)
            data_container.set_expected_outputs(expected_outputs)

        di, self.len_di = self._flatten_list(data_container.data_inputs)
        eo, self.len_eo = self._flatten_list(data_container.expected_outputs)

        flattened_data_container = DataContainer(
            summary_id=data_container.summary_id,
            data_inputs=di,
            expected_outputs=eo,
            sub_data_containers=data_container.sub_data_containers
        )

        return flattened_data_container, context

    def _flatten_list(self, list_to_flatten):
        """
        Flatten the first dimension of a list.

        :param list_to_flatten: list to flatten
        :return: flattened list, len flattened lists
        """
        if not isinstance(list_to_flatten, np.ndarray):
            list_to_flatten = np.array(list_to_flatten)

        if len(list_to_flatten.shape) == 1:
            return list_to_flatten, [1 for x in list_to_flatten]

        list_to_flatten = list(list_to_flatten)
        list_to_flatten = [list(x) for x in list_to_flatten]
        len_list_to_flatten = [len(x) for x in list_to_flatten]
        flattened_list = sum(list_to_flatten, [])

        return flattened_list, len_list_to_flatten

    def _did_process(self, data_container: DataContainer, context: ExecutionContext) -> DataContainer:
        """
        Reaugment the flattened data container.

        :param data_container: data container to then_unflatten
        :param context: execution context
        :return: data container
        """
        data_container = super()._did_process(data_container, context)

        if self.then_unflatten:
            data_container.set_data_inputs(self._reaugment_list(data_container.data_inputs, self.len_di))
            data_container.set_expected_outputs(self._reaugment_list(data_container.expected_outputs, self.len_eo))
            self.len_di = []
            self.len_eo = []

        return data_container

    def _reaugment_list(self, list_to_reaugment, flattened_dimension_lengths):
        """
        Reaugment list with the flattened dimension lengths.

        :param list_to_reaugment: list to then_unflatten
        :return: reaugmented numpy array
        """
        if not self.then_unflatten or list_to_reaugment is None:
            return list_to_reaugment

        reaugmented_list = []
        i = 0
        for list_length in flattened_dimension_lengths:
            sub_list = list_to_reaugment[i:i + list_length]
            reaugmented_list.append(sub_list)
            i += list_length

        return reaugmented_list

    def should_resume(self, data_container: DataContainer, context: ExecutionContext):
        context = context.push(self)

        if isinstance(self.wrapped, ResumableStepMixin) and self.wrapped.should_resume(data_container, context):
            return True
        return False
